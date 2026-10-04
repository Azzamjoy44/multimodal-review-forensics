import os
os.environ.setdefault("USE_TF", "0")        # transformers/optimum: torch only (avoids the protobuf
os.environ.setdefault("USE_TORCH", "1")      # gencode clash) so the text-only Yelp DistilBERT loads
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Request, Form, Response, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from typing import List, Optional
import os
import io
import csv as _csv

from auth import (init_db, register_user, login_user, get_session_user, logout_user,
                  save_upload, get_uploads, delete_item, delete_csv_group)

init_db()

from score_reviews import (
    score_reviews_for_product, score_review_list, score_yelp_review_list,
    _run_distilbert_onnx_sentiment, _run_fake_distilbert_onnx,
    _run_yelp_distilbert,
    _load_model, _load_lstm, _load_fake_lstm,
    _load_distilbert_onnx, _load_fake_distilbert_onnx,
    _load_yelp_lstm, _load_yelp_distilbert, _load_yelp_sklearn_thresholds,
    yelp_dl_variants_available,
    FAKE_MODEL_NAMES, SENTIMENT_MODEL_NAMES, YELP_FAKE_MODEL_NAMES, YELP_MODEL_DIR,
)
from fetch_steam_reviews import fetch_steam_reviews
from load_amazon_reviews import get_reviews_by_product_id, get_reviews_by_user_id, get_dataset_max_time
from load_amazon_reviews import _ensure_loaded as _ensure_amazon_loaded
from load_amazon_labeled import get_labeled_reviews as get_amazon_labeled, get_labeled_stats as get_amazon_labeled_stats
from load_yelp_reviews import (
    get_reviews_by_business_id, get_popular_business_ids, get_test_stats,
    get_reviews_by_user_id as get_yelp_reviews_by_user_id,
)
from load_yelp_reviews import _ensure_loaded as _ensure_yelp_loaded
import review_detector
import model_unified   # UNIFIED CG+human-deception detector — the generalized fake detector used everywhere
                       # (Fake Review Detection browse, user-history chart, and the /analyze box). Replaces Model A.
import model_b   # Model B: multi-modal (text + behavioral) Yelp fake detector for the Yelp section
import csv_behavior   # Tier 3-lite: within-CSV reviewer-behaviour heuristics
from ocr_utils import extract_reviews_from_image, OcrUnavailable, ocr_available

# Folder that holds index.html (and any future CSS/JS files)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load all models and the Amazon index at startup."""
    import asyncio, concurrent.futures
    loop = asyncio.get_event_loop()

    def _preload():
        print("Pre-loading Amazon review index...")
        _ensure_amazon_loaded()
        print("Pre-loading YelpNYC review index...")
        _ensure_yelp_loaded()
        # NOTE: the Amazon-specific fake models (fake_*.joblib, fake_lstm, fake_distilbert)
        # are no longer pre-loaded — the Fake Review Detection section now uses Model A
        # (broadened all-domain detector), same as the Dashboard Analyze box. The Amazon
        # files remain on disk for offline evaluation (evaluate_kaggle.py).
        print("Pre-loading sklearn sentiment models...")
        for name in SENTIMENT_MODEL_NAMES:
            path = os.path.join(os.path.dirname(__file__), "data", f"sentiment_{name}.joblib")
            _load_model(f"sentiment_{name}", path)
        # NOTE: the text-only Yelp baseline models (yelp_fake_*.joblib + yelp LSTM/
        # DistilBERT variants) are no longer pre-loaded — the Yelp section now uses the
        # multi-modal Model B (text + behavioral). The baseline files remain on disk for
        # offline evaluation (evaluate_yelp_models.py / evaluate_yelp_ablation.py).
        print("Pre-loading LSTM ONNX models...")
        _load_lstm()   # sentiment LSTM (Amazon fake LSTM replaced by Model A; Yelp by Model B)
        print("Pre-loading DistilBERT ONNX models...")
        _load_distilbert_onnx()   # sentiment DistilBERT
        print("Pre-loading review/non-review detector models...")
        review_detector.load_all()
        print(f"  review-detector models available: {review_detector.available_models()}")
        print("Pre-loading Model Unified (CG + human-deception) — the generalized fake detector used everywhere...")
        model_unified.preload()
        print("Pre-loading Model B (multi-modal Yelp fake detector)...")
        model_b.preload()
        print("All models loaded — server ready.")

    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, _preload)

    yield


# Create the FastAPI application
app = FastAPI(
    title="Fake Review Detector API",
    description="Loads reviews for a product and returns suspicion scores.",
    version="0.1.0",
    lifespan=lifespan,
)

# Serve everything inside /static at the URL path /static
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
@app.get("/login", include_in_schema=False)
def login_page():
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))


@app.post("/auth/login")
def auth_login(response: Response, username: str = Form(...), password: str = Form(...)):
    token = login_user(username, password)
    if not token:
        return JSONResponse(status_code=401, content={"detail": "Invalid username or password."})
    response = JSONResponse(content={"ok": True})
    response.set_cookie("session_token", token, httponly=True, samesite="lax")
    return response


@app.post("/auth/register")
def auth_register(username: str = Form(...), password: str = Form(...)):
    ok, reason = register_user(username, password)
    if not ok:
        return JSONResponse(status_code=400, content={"detail": reason})
    return JSONResponse(content={"ok": True})


@app.post("/auth/logout")
def auth_logout(request: Request, response: Response):
    token = request.cookies.get("session_token")
    if token:
        logout_user(token)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_token")
    return response


@app.get("/", include_in_schema=False)
def root(request: Request):
    token = request.cookies.get("session_token")
    if not get_session_user(token):
        return RedirectResponse(url="/login", status_code=303)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ---------------------------------------------------------------------------
# GET /gnn-graphs — list the graphs the GNN produced (served from static/gnn/).
# The GNN section of the frontend renders these. Drop any new GNN image into
# static/gnn/ and it appears automatically; add a nicer title/caption below.
# ---------------------------------------------------------------------------
GNN_DIR = os.path.join(STATIC_DIR, "gnn")
_GNN_META = {
    "yelp_rings_gnn_corrected_viz.png": {
        "title": "Yelp — Coordinated fake-review rings (GNN)",
        "caption": "Reviewers (circles) converging on the businesses they jointly target "
                   "(blue squares); edges are the GNN-flagged reviews. Reviewer colour is the "
                   "ground-truth check: red = truly fake (correct catch), grey = false positive. "
                   "Detected label-free from the corrected reviewer↔review↔business graph with "
                   "reviewer↔reviewer co-edges.",
    },
}


@app.get("/gnn-graphs")
def gnn_graphs():
    items = []
    if os.path.isdir(GNN_DIR):
        for fn in sorted(os.listdir(GNN_DIR)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".webp")):
                meta = _GNN_META.get(fn, {})
                items.append({
                    "url":     f"/static/gnn/{fn}",
                    "title":   meta.get("title", fn.rsplit(".", 1)[0].replace("_", " ")),
                    "caption": meta.get("caption", ""),
                })
    return {"graphs": items}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /reviews
#
# Two modes — supply exactly one of the two query parameters:
#
#   Sample mode  →  ?product_name=Cyberpunk+2077
#                   Loads from data/sample_reviews.json and scores them.
#
#   Live mode    →  ?app_id=1091500
#                   Fetches live reviews from Steam, then scores them.
#                   Optional: add &count=20 to control how many are fetched
#                   (default 20, max 100).
#
# Both modes return the same JSON shape so the frontend never needs changing.
# ---------------------------------------------------------------------------
def _score_reviews(raw_reviews, task):
    """Score a list of reviews for the Fake Review Detection / Sentiment sections.

    For task='fake' the UNIFIED generalized detector (CG + human-deception) supplies the
    fake verdicts + word/sentence highlights — the same detector used by the Dashboard
    Analyze box — instead of the Amazon-only fake models. Rule-based suspicion and
    sentiment are unchanged. This also keeps the Amazon-specific fake models out of RAM
    (they're retained on disk for offline evaluation only)."""
    if task == "fake":
        texts = [r.get("review_text", "") for r in raw_reviews]
        return score_review_list(raw_reviews, task="fake", fake_override=model_unified.score_fake(texts))
    return score_review_list(raw_reviews, task=task)


@app.get("/reviews")
def get_reviews(
    product_name: str = Query(
        default=None,
        description="Sample mode: name of the product in the local JSON file",
    ),
    app_id: int = Query(
        default=None,
        description="Live mode: numeric Steam app ID (e.g. 1091500)",
    ),
    product_id: str = Query(
        default=None,
        description="Amazon mode: ProductId from Reviews.csv (e.g. B001E4KFG0)",
    ),
    count: int = Query(
        default=20,
        ge=1,
        le=500,
        description="Number of reviews to return (Amazon and Steam modes)",
    ),
    task: str = Query(
        default="fake",
        description="Which pipeline to run: 'fake' for fake detection, 'sentiment' for sentiment analysis",
    ),
):
    # ------------------------------------------------------------------ #
    # Guard: at least one parameter must be supplied                       #
    # ------------------------------------------------------------------ #
    if app_id is None and product_name is None and product_id is None:
        raise HTTPException(
            status_code=400,
            detail="Provide one of: 'product_name' (sample), 'app_id' (Steam), or 'product_id' (Amazon).",
        )

    # ------------------------------------------------------------------ #
    # AMAZON MODE — product_id was supplied                                #
    # ------------------------------------------------------------------ #
    if product_id is not None:
        raw_reviews = get_reviews_by_product_id(product_id, limit=count)

        if not raw_reviews:
            raise HTTPException(
                status_code=404,
                detail=f"No reviews found for Amazon product_id='{product_id}'. "
                       "Check the product ID and try again.",
            )

        scored = _score_reviews(raw_reviews, task)

        return {
            "mode":          "amazon",
            "product_id":    product_id,
            "total_reviews": len(scored),
            "reviews":       scored,
        }

    # ------------------------------------------------------------------ #
    # LIVE MODE — app_id was supplied                                      #
    # ------------------------------------------------------------------ #
    if app_id is not None:
        # Fetch and normalise reviews from Steam
        raw_reviews = fetch_steam_reviews(app_id, count=count)

        if not raw_reviews:
            raise HTTPException(
                status_code=502,    # 502 = upstream (Steam) gave us nothing useful
                detail=f"Could not fetch reviews from Steam for app_id={app_id}. "
                       "Check the app ID and try again.",
            )

        # Score the live reviews using the shared scoring function
        scored = _score_reviews(raw_reviews, task)

        # Use the product name that the fetcher resolved (e.g. "Cyberpunk 2077")
        resolved_name = raw_reviews[0].get("product_name", f"steam_app_{app_id}")

        return {
            "mode":          "live",
            "product_name":  resolved_name,
            "app_id":        app_id,
            "total_reviews": len(scored),
            "reviews":       scored,
        }

    # ------------------------------------------------------------------ #
    # SAMPLE MODE — product_name was supplied                              #
    # ------------------------------------------------------------------ #
    scored = score_reviews_for_product(product_name)

    if not scored:
        raise HTTPException(
            status_code=404,
            detail=f"No reviews found for product: '{product_name}'",
        )

    return {
        "mode":          "sample",
        "product_name":  product_name,
        "total_reviews": len(scored),
        "reviews":       scored,
    }


# ---------------------------------------------------------------------------
# GET /amazon-fake-labeled
# Labeled Amazon fake-review browse (held-out products test split: real reviews +
# AI-generated fakes), scored by the unified detector, with ground_truth_label so the
# Fake Review Detection UI can show predicted-vs-actual (like the Yelp section).
# ---------------------------------------------------------------------------
@app.get("/amazon-fake-labeled")
def amazon_fake_labeled(
    count: int = Query(default=15, ge=1, le=200, description="number of labeled reviews"),
    offset: int = Query(default=0, ge=0, description="skip the first N (incremental loading)"),
    dataset: str = Query(default="indist", description="'indist' (held-out Amazon) | 'ood' (HC3, never trained on)"),
    stats: bool = Query(default=False, description="return dataset stats instead of reviews"),
):
    if stats:
        return get_amazon_labeled_stats(dataset)
    raw = get_amazon_labeled(count, offset, dataset)
    if not raw:
        raise HTTPException(404, f"No labeled reviews for dataset='{dataset}'. Run the matching build script.")
    scored = _score_reviews(raw, "fake")
    for sc, r in zip(scored, raw):           # carry ground truth + source onto the scored output
        sc["ground_truth_label"] = r["ground_truth_label"]
        sc["source"] = r.get("source", "")
    return {"mode": "amazon-labeled", "dataset": dataset, "total_reviews": len(scored), "reviews": scored}


# ---------------------------------------------------------------------------
# GET /yelp-reviews
#
# Returns Yelp reviews for a given business, scored by the Yelp-specific
# fake-detection models (trained on YelpZip + YelpChi + YelpNYC).
# Also exposes a ?popular=true shortcut that returns the 10 most-reviewed
# business IDs so the frontend can populate a default picker.
# ---------------------------------------------------------------------------
@app.get("/yelp-reviews")
def get_yelp_reviews(
    business_id: str = Query(
        default=None,
        description="Prefixed business ID (e.g. 'zip_0', 'nyc_42', 'chi_res_abc123')",
    ),
    count: int = Query(
        default=20,
        ge=1,
        le=200,
        description="Number of reviews to return",
    ),
    popular: bool = Query(
        default=False,
        description="If true, return the 10 most-reviewed business IDs instead of reviews",
    ),
    stats: bool = Query(
        default=False,
        description="If true, return dataset stats (total, fake, genuine, businesses)",
    ),
    task: str = Query(
        default="fake",
        description="'fake' for Yelp fake-detection models, 'sentiment' for standard sentiment models",
    ),
    variant: str = Query(
        default="baseline",
        description="text-only DL variant (only used when model='baseline'): 'baseline' | 'focal' | 'contrastive'",
    ),
    model: str = Query(
        default="modelb",
        description="which detector to score with: 'modelb' (multi-modal, standard DL) | "
                    "'modelb_focal' | 'modelb_contrastive' (Model B with focal/contrastive DL members) | "
                    "'baseline' (text-only, uses 'variant')",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="skip the first N reviews (for the frontend's incremental cache: fetch only newly-needed reviews)",
    ),
    variants: bool = Query(
        default=False,
        description="If true, return the list of available DL model variants",
    ),
):
    if variants:
        return {"variants": yelp_dl_variants_available()}

    if stats:
        return get_test_stats()

    if popular:
        return {"popular_business_ids": get_popular_business_ids(10)}

    if business_id is None:
        raise HTTPException(
            status_code=400,
            detail="Provide 'business_id', or set 'popular=true' / 'stats=true'.",
        )

    raw_reviews = get_reviews_by_business_id(business_id, limit=count, offset=offset)
    if not raw_reviews:
        raise HTTPException(
            status_code=404,
            detail=f"No test-split reviews found for business_id='{business_id}'.",
        )

    if task == "sentiment":
        scored = score_review_list(raw_reviews, task="sentiment")
        return {
            "mode":          "yelp-sentiment",
            "business_id":   business_id,
            "total_reviews": len(scored),
            "reviews":       scored,
        }

    # model='modelb' (default): multi-modal Model B (text + behavioral) supplies the fake
    # verdicts + highlights. model='baseline': the text-only Yelp models (8 sklearn + the
    # chosen LSTM/DistilBERT `variant`), so the UI can compare the two detectors head-to-head.
    # cache-aware duplicate detection: when paging in NEW reviews (offset>0), pass the already-
    # loaded reviews as context so a new review duplicating a cached one is still flagged.
    dup_context = get_reviews_by_business_id(business_id, limit=offset, offset=0) if offset else None
    if model == "baseline":
        scored = score_yelp_review_list(raw_reviews, variant=variant, dup_context=dup_context)
    else:
        # model ∈ {modelb, modelb_focal, modelb_contrastive} → Model B; the suffix selects which
        # DL members (LSTM+DistilBERT) score — the 8 multi-modal sklearn are shared across variants.
        mb_variant = {"modelb": "standard", "modelb_focal": "focal",
                      "modelb_contrastive": "contrastive"}.get(model, "standard")
        bundles = model_b.score_yelp_fake(raw_reviews, variant=mb_variant)
        scored  = score_yelp_review_list(raw_reviews, fake_override=bundles, dup_context=dup_context)
    return {
        "mode":          "yelp",
        "business_id":   business_id,
        "model":         model,
        "variant":       variant,
        "total_reviews": len(scored),
        "reviews":       scored,
    }


# ---------------------------------------------------------------------------
# GET /user-history
# Returns review history and activity stats for a given Amazon UserId.
# ---------------------------------------------------------------------------
@app.get("/user-history")
def get_user_history(user_id: str = Query(..., description="Amazon UserId")):
    reviews = get_reviews_by_user_id(user_id)

    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=f"No reviews found for user '{user_id}'.",
        )

    dataset_now   = get_dataset_max_time()
    six_months_s  = 6 * 30 * 24 * 3600

    last_6m = [r for r in reviews if dataset_now - r["time"] <= six_months_s]
    per_day  = round(len(last_6m) / 180, 2) if last_6m else 0

    # Products reviewed more than once
    from collections import Counter, defaultdict
    product_counts = Counter(r["product_id"] for r in reviews)
    duplicates     = [
        {"product_id": pid, "review_count": cnt}
        for pid, cnt in product_counts.items() if cnt > 1
    ]

    # Identical review texts posted across different products
    text_groups = defaultdict(list)
    for r in reviews:
        key = r["review_text"].strip().lower()
        if key:
            text_groups[key].append(r)
    identical_texts = [
        {
            "review_text": group[0]["review_text"],
            "summary":     group[0]["summary"],
            "products":    [r["product_id"] for r in group],
            "dates":       [r["time"] for r in group],
            "count":       len(group),
        }
        for group in text_groups.values() if len(group) > 1
    ]

    # Identical summaries posted across different products (even if text differs)
    summary_groups = defaultdict(list)
    for r in reviews:
        key = r["summary"].strip().lower()
        if key:
            summary_groups[key].append(r)
    identical_summaries = [
        {
            "summary":  group[0]["summary"],
            "products": [r["product_id"] for r in group],
            "count":    len(group),
        }
        for group in summary_groups.values()
        if len(group) > 1 and group[0]["summary"].strip()
        # only report if not already covered by identical_texts
        and group[0]["review_text"].strip().lower() not in text_groups
    ]

    return {
        "user_id":               user_id,
        "total_reviews":         len(reviews),
        "reviews_last_6_months": len(last_6m),
        "reviews_per_day":       per_day,
        "duplicate_products":    duplicates,
        "identical_texts":       identical_texts,
        "identical_summaries":   identical_summaries,
        "reviews": (lambda sent_preds, fake_preds: [
            {
                "review_id":      r["review_id"],
                "product_id":     r["product_id"],
                "rating":         r["rating"],
                "time":           r["time"],
                "summary":        r["summary"],
                "review_text":    r["review_text"],
                "sentiment":      sent_preds[i].get("label"),  # 0=negative, 1=positive
                "fake_predicted": fake_preds[i].get("label"),  # 0=genuine, 1=fake
            }
            for i, r in enumerate(reviews)
        ])(
            _run_distilbert_onnx_sentiment([r["review_text"] for r in reviews]),
            model_unified.score_fake_distilbert([r["review_text"] for r in reviews]),  # unified DistilBERT (same as Fake/Analyze)
        ),
    }


# ---------------------------------------------------------------------------
# GET /yelp-user-history
# Returns review history for a given Yelp user_id, scored by the Yelp
# DistilBERT fake-detection model.
# ---------------------------------------------------------------------------
@app.get("/yelp-user-history")
def get_yelp_user_history(user_id: str = Query(..., description="Yelp user ID")):
    from collections import Counter, defaultdict

    reviews = get_yelp_reviews_by_user_id(user_id)
    if not reviews:
        raise HTTPException(
            status_code=404,
            detail=f"No Yelp reviews found for user '{user_id}'.",
        )

    texts       = [r["review_text"] for r in reviews]
    fake_preds  = model_b.score_yelp_distilbert(reviews)   # multi-modal DistilBERT (text + behavioral)
    sent_preds  = _run_distilbert_onnx_sentiment(texts)

    # Duplicate businesses
    biz_counts   = Counter(r["business_id"] for r in reviews)
    dup_biz      = [{"business_id": bid, "review_count": cnt}
                    for bid, cnt in biz_counts.items() if cnt > 1]

    # Identical texts across different businesses
    text_groups: dict = defaultdict(list)
    for r in reviews:
        key = r["review_text"].strip().lower()
        if key:
            text_groups[key].append(r)
    identical_texts = [
        {
            "review_text": group[0]["review_text"],
            "businesses":  [r["business_id"] for r in group],
            "dates":       [r["time"] for r in group],
            "count":       len(group),
        }
        for group in text_groups.values() if len(group) > 1
    ]

    return {
        "user_id":             user_id,
        "total_reviews":       len(reviews),
        "duplicate_businesses": dup_biz,
        "identical_texts":     identical_texts,
        "reviews": [
            {
                "review_id":          r["review_id"],
                "business_id":        r["business_id"],
                "source":             r["source"],
                "rating":             r["rating"],
                "time":               r["time"],
                "review_text":        r["review_text"],
                "ground_truth_label": r["ground_truth_label"],
                "fake_predicted":     fake_preds[i].get("label"),
                "sentiment":          sent_preds[i].get("label"),
            }
            for i, r in enumerate(reviews)
        ],
    }


# ---------------------------------------------------------------------------
# POST /analyze  — Dashboard "Analyze" feature
#
# Accepts any combination of: a single `text`, a `csv_file`, and image file(s).
# Each provided text is first sent through the review/non-review GATE; only texts
# the majority of models call a review get fake-detection + sentiment analysis.
# CSV: <50% review rows -> error; >=50% -> analyze the review rows, warn on the rest.
# ---------------------------------------------------------------------------
def _merge_fake_sentiment(texts):
    """Fake detection via the UNIFIED generalized ensemble + sentiment, merged per text.

    Tier 1: uploaded reviews are scored by the unified CG+human-deception detector (8 sklearn
    + LSTM + DistilBERT, trained on machine-generated AND human-written deceptive reviews) —
    the true generalized fake detector, so it catches both AI fakes and human deception across
    any review type. Sentiment still comes from the general sentiment models.
    """
    fake = model_unified.score_fake(texts)   # rich: fake_models + phrase/sentence highlight data
    reviews = [{"review_id": f"item_{i}", "review_text": t, "rating": 3}
               for i, t in enumerate(texts)]
    sent = score_review_list(reviews, task="sentiment")
    out = []
    for i in range(len(texts)):
        d = dict(fake[i])
        s = sent[i]
        d["sentiment_models"] = s["sentiment_models"]
        # carry the sentiment highlight lists so the Analyze card can highlight on click
        d["sentiment_positive_phrases"]      = s.get("sentiment_positive_phrases", [])
        d["sentiment_negative_phrases"]      = s.get("sentiment_negative_phrases", [])
        d["sentiment_dl_positive_sentences"] = s.get("sentiment_dl_positive_sentences", [])
        d["sentiment_dl_negative_sentences"] = s.get("sentiment_dl_negative_sentences", [])
        out.append(d)
    return out


def _detect_batch_duplicates(texts, near_threshold=0.85, near_cap=3000):
    """Tier-2 human-fake signal: find exact + near-duplicate reviews WITHIN an upload.

    Coordinated/planted reviews are often copy-pasted or lightly reworded across an
    upload — a signal we CAN compute from text alone (unlike polished human fakes).
    Exact dups via normalized-text grouping (O(n)); near dups via TF-IDF cosine
    (only for modest batches, to stay fast).
    Returns one dict per text: {is_duplicate, type: exact|near|None, matches:[idx], max_similarity}.
    """
    import re as _re
    n = len(texts)
    dup = [{"is_duplicate": False, "type": None, "matches": [], "max_similarity": 0.0}
           for _ in range(n)]
    norms = [_re.sub(r"\s+", " ", (t or "").strip().lower()) for t in texts]

    # exact duplicates (ignore trivially short text)
    groups = {}
    for i, nm in enumerate(norms):
        if nm and len(nm.split()) >= 3:
            groups.setdefault(nm, []).append(i)
    for idxs in groups.values():
        if len(idxs) > 1:
            for i in idxs:
                dup[i].update(is_duplicate=True, type="exact", max_similarity=1.0,
                              matches=[j for j in idxs if j != i])

    # near duplicates (TF-IDF cosine) — only on a modest, non-trivial batch
    valid = [i for i in range(n) if len(norms[i].split()) >= 5]
    if 2 <= len(valid) <= near_cap:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            X = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform([norms[i] for i in valid])
            sim = cosine_similarity(X)
            for a in range(len(valid)):
                for b in range(a + 1, len(valid)):
                    s = float(sim[a, b])
                    if s >= near_threshold:
                        for x, y in ((valid[a], valid[b]), (valid[b], valid[a])):
                            if dup[x]["type"] != "exact":
                                dup[x]["is_duplicate"] = True
                                dup[x]["type"] = dup[x]["type"] or "near"
                                if y not in dup[x]["matches"]:
                                    dup[x]["matches"].append(y)
                                dup[x]["max_similarity"] = max(dup[x]["max_similarity"], s)
        except Exception:
            pass
    return dup


def _analyze_texts(texts, sources=None):
    """Gate each text; attach fake+sentiment analysis to the ones that are reviews."""
    texts = [("" if t is None else str(t)) for t in texts]
    det = review_detector.classify_texts(texts)
    review_idx = [i for i, d in enumerate(det) if d["is_review"]]
    scored = {}
    if review_idx:
        merged = _merge_fake_sentiment([texts[i] for i in review_idx])
        for j, i in enumerate(review_idx):
            scored[i] = merged[j]
    dups = _detect_batch_duplicates(texts)   # Tier-2 coordinated-fake signal
    items = []
    for i, t in enumerate(texts):
        d = det[i]
        dup = dups[i]
        dup_matches = [(sources[j] if sources else f"#{j + 1}") for j in dup["matches"]]
        items.append({
            "source":            (sources[i] if sources else "text"),
            "text":              t,
            "is_review":         d["is_review"],
            "review_votes":      d["review_votes"],
            "total_models":      d["total_models"],
            "dissenting_models": d["dissenting_models"],
            "review_detection": {                   # Review Detection row: per-model votes + highlights
                "per_model":            d.get("per_model", {}),
                "review_phrases":       d.get("review_phrases", []),
                "nonreview_phrases":    d.get("nonreview_phrases", []),
                "dl_review_sentences":  d.get("dl_review_sentences", []),
                "dl_nonreview_sentences": d.get("dl_nonreview_sentences", []),
            },
            "analysis":          scored.get(i),   # None when not a review
            "duplicate": {                          # Tier-2: duplicate within this upload
                "is_duplicate":   dup["is_duplicate"],
                "type":           dup["type"],       # "exact" | "near" | None
                "matches":        dup_matches,       # which other uploaded rows it matches
                "max_similarity": round(dup["max_similarity"], 3),
            },
        })
    return items


def _analyze_csv(rows, filename):
    if not rows:
        return {"filename": filename, "status": "error", "message": "The CSV file is empty."}
    header = [h.strip().lower() for h in rows[0]]
    text_col, data_rows = None, rows[1:]
    for cand in ("review_text", "text", "review", "content", "body", "comment"):
        if cand in header:
            text_col = header.index(cand); break
    if text_col is None:
        if len(rows[0]) == 1:          # single unnamed column -> treat all rows as data
            text_col, data_rows = 0, rows
        else:
            return {"filename": filename, "status": "error",
                    "message": "No text column found. Add a column named 'text' or 'review'."}
    texts = [(r[text_col] if text_col < len(r) else "").strip() for r in data_rows]
    if not any(texts):
        return {"filename": filename, "status": "error",
                "message": "No text found in the CSV's text column."}

    items   = _analyze_texts(texts, sources=[f"row {i + 1}" for i in range(len(texts))])

    # Tier 3-lite: within-CSV reviewer-behaviour heuristics (if the CSV has a user column).
    # Catches coordinated/astroturf reviewers from patterns visible in the upload itself
    # (prolific reviewers, all-extreme/identical ratings, same-day bursts).
    colmap = csv_behavior.detect_columns(rows[0])
    if colmap:
        for it, beh in zip(items, csv_behavior.from_rows(data_rows, colmap)):
            it["behavioral"] = beh

    total   = len(items)
    n_review = sum(1 for it in items if it["is_review"])
    pct     = (n_review / total * 100) if total else 0

    if pct < 50:
        for it in items:                 # spec: under 50% -> analyze nothing
            it["analysis"] = None
        return {"filename": filename, "status": "error",
                "rows": total, "review_rows": n_review, "review_pct": round(pct, 1),
                "message": (f"Only {n_review}/{total} rows ({pct:.0f}%) were identified as "
                            "reviews by the majority of models — under 50%, so none were analyzed."),
                "items": items}
    warning = None
    if n_review < total:
        warning = (f"{total - n_review} of {total} rows were not identified as reviews by "
                   "the majority and were skipped.")
    return {"filename": filename, "status": "ok", "warning": warning,
            "rows": total, "review_rows": n_review, "review_pct": round(pct, 1), "items": items}


@app.post("/analyze")
async def analyze(
    request: Request,
    text: Optional[str] = Form(default=None),
    csv_file: Optional[UploadFile] = File(default=None),
    images: Optional[List[UploadFile]] = File(default=None),
):
    if not review_detector.available_models():
        raise HTTPException(503, "Review-detector models are not trained/loaded yet.")

    provided_text = bool(text and text.strip())
    has_csv       = csv_file is not None
    has_images    = bool(images)
    if not (provided_text or has_csv or has_images):
        raise HTTPException(400, "Provide at least one of: text, a CSV file, or image(s).")

    out = {"text": None, "images": [], "csv": None}

    if provided_text:
        out["text"] = _analyze_texts([text.strip()], sources=["text"])[0]

    if has_images:
        if not ocr_available():
            out["images_error"] = ("Image OCR is not configured. Set the MISTRAL_API_KEY "
                                    "environment variable and `pip install mistralai`, then "
                                    "restart the server.")
        else:
            # Mistral vision OCR: each image -> a list of CLEAN, separated review bodies.
            # A multi-review screenshot therefore becomes multiple independent reviews.
            texts, names = [], []
            for img in images:
                data = await img.read()
                base = img.filename or "image"
                try:
                    reviews = extract_reviews_from_image(data, img.content_type or "image/png")
                except OcrUnavailable as e:
                    out["images_error"] = str(e); break
                except Exception as e:
                    out["images_error"] = f"OCR failed on {base}: {e}"; reviews = []
                reviews = [r for r in reviews if r.strip()]
                if len(reviews) == 1:
                    texts.append(reviews[0]); names.append(base)
                else:                                   # 0 or many -> tag each with its index
                    for k, rv in enumerate(reviews, 1):
                        texts.append(rv); names.append(f"{base} · review {k}")
            if texts:
                items = _analyze_texts(texts, sources=names)
                for it, nm in zip(items, names):
                    it["filename"] = nm
                out["images"] = items
            elif "images_error" not in out:
                out["images_error"] = "No review text was found in the uploaded image(s)."

    if has_csv:
        raw  = (await csv_file.read()).decode("utf-8", "replace")
        rows = list(_csv.reader(io.StringIO(raw)))
        out["csv"] = _analyze_csv(rows, csv_file.filename or "file.csv")

    # Persist this request's results for the logged-in user's "My Uploads" view.
    # (Mutates the items in `out` to attach review_id + uploaded_at, which is harmless.)
    username = get_session_user(request.cookies.get("session_token", ""))
    if username:
        try:
            save_upload(username, out.get("text"), out.get("images"), out.get("csv"))
        except Exception as e:                       # never let persistence break analysis
            print(f"[my-uploads] save failed for {username}: {e}")

    return out


@app.get("/personal-results")
async def personal_results(request: Request):
    """The logged-in user's saved Analyze results, for the My Uploads view."""
    username = get_session_user(request.cookies.get("session_token", ""))
    if not username:
        raise HTTPException(401, "Not logged in.")
    return get_uploads(username)


@app.post("/personal-results/delete-item")
async def personal_delete_item(request: Request):
    """Delete one saved text/image review result (by review_id) for the current user."""
    username = get_session_user(request.cookies.get("session_token", ""))
    if not username:
        raise HTTPException(401, "Not logged in.")
    body = await request.json()
    review_id = (body or {}).get("review_id", "")
    if not review_id:
        raise HTTPException(400, "review_id is required.")
    return {"deleted": delete_item(username, review_id)}


@app.post("/personal-results/delete-csv")
async def personal_delete_csv(request: Request):
    """Delete a whole saved CSV upload (all its review results) for the current user."""
    username = get_session_user(request.cookies.get("session_token", ""))
    if not username:
        raise HTTPException(401, "Not logged in.")
    body = await request.json()
    batch_id = (body or {}).get("batch_id", "")
    if not batch_id:
        raise HTTPException(400, "batch_id is required.")
    return {"deleted": delete_csv_group(username, batch_id)}

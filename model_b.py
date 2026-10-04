"""
model_b.py — the MULTI-MODAL Yelp fake detector ("Model B") for the live Yelp Fake
Review Detection section (and Yelp user-history chart).

Unlike the text-only Yelp baseline (which plateaus at 66-69% macro-F1 because Yelp's
labels are behavioral), Model B FUSES the review text with 17 engineered behavioral
features (reviewer/business metadata aggregated over the full 1.03M-review pool). The
features are carried on each review by load_yelp_reviews.py (from
data/yelp_frontend_multimodal.csv).

Three families, all multi-modal (text + behavioral), matching the frontend dot-strip.
SERVED = the **pure-full** models (trained on the full deduped 678k Yelp, NO AI augmentation;
~77% macro-F1, and verified to beat text-only on the corrected frontend set — see THESIS_NOTES
"Corrected frontend comparison"). The earlier AI-augmented dirs (yelp_multimodal_{sklearn_models,
lstm_onnx,distilbert_onnx}) are retained on disk for reference but no longer served.
    data/yelp_multimodal_full_sklearn_models/     8 sklearn Pipelines (ColumnTransformer:
                                                  TF-IDF on text + MinMaxScaler on features)
    data/yelp_multimodal_lstm_full_onnx/          BiLSTM text branch + MLP behavioral branch
                                                  (ONNX; text_input[250] + beh_input[17])
    data/yelp_multimodal_distilbert_full_onnx/    DistilBERT text + behavioral
                                                  (ONNX; input_ids/attention_mask[256] + beh[17])

Output per review is the same `fake_models` shape the frontend renders + the phrase/
sentence highlight lists, so it drops into score_yelp_review_list via fake_override.
"""

import os
import json

import numpy as np

# reuse the browse-card phrase extractors (data-driven from per-model reasons)
from score_reviews import _get_suspicious_phrases, _get_genuine_phrases, _split_sentences

_DATA   = os.path.join(os.path.dirname(__file__), "data")
# SERVED = pure-full models (no AI; ~77%, beats text-only on the corrected frontend set).
SK_DIR  = os.path.join(_DATA, "yelp_multimodal_full_sklearn_models")
LSTM_DIR = os.path.join(_DATA, "yelp_multimodal_lstm_full_onnx")
DB_DIR  = os.path.join(_DATA, "yelp_multimodal_distilbert_full_onnx")
SK_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]

# behavioral feature order — identical for sklearn (ColumnTransformer 'beh' cols),
# LSTM and DistilBERT (their meta feat_cols). user_reviews_on_this_biz is NOT used.
FEAT_COLS = [
    "rating", "is_extreme", "review_word_len", "review_char_len",
    "rating_dev_from_biz", "abs_rating_dev_from_biz",
    "user_review_count", "user_avg_rating", "user_rating_std",
    "user_frac_positive", "user_frac_extreme", "user_is_singleton",
    "user_max_reviews_per_day", "user_reviews_per_day",
    "biz_review_count", "biz_avg_rating", "biz_rating_std",
]
LSTM_MAX_LEN = 250
DB_MAX_LEN   = 256

# DL-member VARIANTS. The standard Model B serves the pure-full DL members; the focal/contrastive
# variants swap ONLY the LSTM + DistilBERT members (same multi-modal architecture, only the loss
# differs — see THESIS_NOTES "Model B focal/contrastive DL variants"). The 8 multi-modal sklearn
# are SHARED across variants (there are no focal/contrastive sklearn). Tuned best-shot configs are
# baked into each folder's meta. The variant ONNX are drop-in: identical input/output names.
_VARIANT_LSTM_DIRS = {
    "standard":    LSTM_DIR,
    "focal":       os.path.join(_DATA, "yelp_multimodal_lstm_focal_onnx"),
    "contrastive": os.path.join(_DATA, "yelp_multimodal_lstm_contrastive_onnx"),
}
_VARIANT_DB_DIRS = {
    "standard":    DB_DIR,
    "focal":       os.path.join(_DATA, "yelp_multimodal_distilbert_focal_onnx"),
    "contrastive": os.path.join(_DATA, "yelp_multimodal_distilbert_contrastive_onnx"),
}


def yelp_modelb_variants_available():
    """Model-B DL variants whose LSTM *and* DistilBERT folders both exist on disk
    (the frontend offers only selectable ones). 'standard' is always first."""
    avail = []
    for v in ("standard", "focal", "contrastive"):
        if os.path.isdir(_VARIANT_LSTM_DIRS[v]) and os.path.isdir(_VARIANT_DB_DIRS[v]):
            avail.append(v)
    return avail or ["standard"]


# ---------------------------------------------------------------------------
# Pure-Python Keras tokenizer (mirrors model_a / score_reviews)
# ---------------------------------------------------------------------------
class _KerasTokenizer:
    def __init__(self, word_index, filters, lower, num_words):
        self.word_index = word_index
        self._filters = str.maketrans("", "", filters)
        self.lower = lower
        self.num_words = num_words

    @classmethod
    def from_json(cls, path):
        cfg = json.load(open(path, encoding="utf-8"))["config"]
        wi = cfg["word_index"]
        if isinstance(wi, str):
            wi = json.loads(wi)
        return cls(wi, cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
                   cfg.get("lower", True), cfg.get("num_words"))

    def texts_to_sequences(self, texts):
        seqs = []
        for text in texts:
            if self.lower:
                text = text.lower()
            text = text.translate(self._filters)
            seq = []
            for w in text.split():
                idx = self.word_index.get(w)
                if idx is not None and (self.num_words is None or idx < self.num_words):
                    seq.append(idx)
            seqs.append(seq)
        return seqs


def _pad(seqs, maxlen):
    out = np.zeros((len(seqs), maxlen), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s[:maxlen]
        out[i, :len(s)] = s
    return out


# ---------------------------------------------------------------------------
# Lazy-loaded caches
# ---------------------------------------------------------------------------
_sk = None; _sk_thr = None
_lstm = {}        # variant -> (sess, tokenizer, scaler, threshold) | False
_db = {}          # variant -> (sess, tokenizer, scaler, threshold) | False

# Per-review SKLEARN cache, keyed by review_id. The 8 multi-modal sklearn are SHARED across the
# standard/focal/contrastive variants, so once a review has been sklearn-scored under any variant
# its result is reused for the others — switching variants (or paging in more reviews) recomputes
# sklearn ONLY for reviews not seen before. The DL members differ per variant and are NOT cached
# here (always computed by the caller). Bounded to avoid unbounded growth.
_sk_review_cache: dict = {}
_SK_CACHE_MAX = 25000


def _load_sklearn():
    global _sk, _sk_thr
    if _sk is not None:
        return _sk, _sk_thr
    import joblib
    _sk, _sk_thr = {}, {}
    thr_path = os.path.join(SK_DIR, "thresholds.json")
    thresholds = json.load(open(thr_path)) if os.path.exists(thr_path) else {}
    for n in SK_NAMES:
        p = os.path.join(SK_DIR, f"{n}.joblib")
        if os.path.exists(p):
            try:
                _sk[n] = joblib.load(p); _sk_thr[n] = float(thresholds.get(n, 0.5))
            except Exception as e:
                print(f"Model B: sklearn {n} load failed — {e}")
    print(f"Model B: loaded {len(_sk)} multimodal sklearn models.")
    return _sk, _sk_thr


def _resolve_variant_dir(dirs, variant):
    """Variant dir, falling back to 'standard' if the requested one is missing on disk."""
    d = dirs.get(variant, dirs["standard"])
    return d if os.path.isdir(d) else dirs["standard"]


def _load_lstm(variant="standard"):
    if variant in _lstm:
        return _lstm[variant]
    d = _resolve_variant_dir(_VARIANT_LSTM_DIRS, variant)
    onnx = os.path.join(d, "yelp_multimodal_lstm.onnx")
    tok  = os.path.join(d, "yelp_multimodal_lstm_tokenizer.json")
    scl  = os.path.join(d, "yelp_multimodal_lstm_scaler.joblib")
    meta = os.path.join(d, "yelp_multimodal_lstm_meta.json")
    if os.path.exists(onnx) and os.path.exists(tok) and os.path.exists(scl):
        try:
            import onnxruntime as ort, joblib
            sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
            thr = json.load(open(meta)).get("threshold", 0.5) if os.path.exists(meta) else 0.5
            _lstm[variant] = (sess, _KerasTokenizer.from_json(tok), joblib.load(scl), float(thr))
            print(f"Model B: multimodal LSTM ONNX loaded ({variant}).")
        except Exception as e:
            print(f"Model B: LSTM load failed ({variant}) — {e}"); _lstm[variant] = False
    else:
        _lstm[variant] = False
    return _lstm[variant]


def _load_db(variant="standard"):
    if variant in _db:
        return _db[variant]
    d = _resolve_variant_dir(_VARIANT_DB_DIRS, variant)
    onnx = os.path.join(d, "yelp_multimodal_distilbert.onnx")
    scl  = os.path.join(d, "yelp_multimodal_distilbert_scaler.joblib")
    meta = os.path.join(d, "yelp_multimodal_distilbert_meta.json")
    # tokenizer may sit at the dir root (standard) or in a tokenizer/ subdir (variant zips)
    tok_dir = os.path.join(d, "tokenizer") if os.path.isdir(os.path.join(d, "tokenizer")) else d
    if os.path.exists(onnx) and os.path.exists(scl):
        try:
            import onnxruntime as ort, joblib
            from transformers import AutoTokenizer
            sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
            tok = AutoTokenizer.from_pretrained(tok_dir)
            thr = json.load(open(meta)).get("threshold", 0.5) if os.path.exists(meta) else 0.5
            _db[variant] = (sess, tok, joblib.load(scl), float(thr))
            print(f"Model B: multimodal DistilBERT ONNX loaded ({variant}).")
        except Exception as e:
            print(f"Model B: DistilBERT load failed ({variant}) — {e}"); _db[variant] = False
    else:
        _db[variant] = False
    return _db[variant]


def preload():
    """Eager-load the STANDARD Model-B members (call at server startup).
    Focal/contrastive DL variants lazy-load on first request to keep startup RAM down."""
    _load_sklearn(); _load_lstm("standard"); _load_db("standard")


# ---------------------------------------------------------------------------
def _conf(p, label):
    return round(float(p if label == 1 else 1 - p) * 100)


def _texts(reviews):
    return [str(r.get("review_text", "") or "") for r in reviews]


def _feat_matrix(reviews):
    """[N, 17] raw behavioral features in FEAT_COLS order (unscaled)."""
    M = np.zeros((len(reviews), len(FEAT_COLS)), dtype=np.float32)
    for i, r in enumerate(reviews):
        for j, c in enumerate(FEAT_COLS):
            v = r.get(c, 0.0)
            try:
                M[i, j] = float(v) if v not in ("", None) else 0.0
            except (ValueError, TypeError):
                M[i, j] = 0.0
    return M


def _none(reviews, msg):
    return [{"label": None, "confidence": None, "reasons": [msg]} for _ in reviews]


def _compute_sklearn(reviews):
    """Run the 8 multi-modal sklearn over `reviews` (no caching) → per-review {model: {...}}."""
    sk, thr = _load_sklearn()
    if not sk:
        return [{} for _ in reviews]
    import pandas as pd
    texts = _texts(reviews)
    df = pd.DataFrame({"text": texts})
    M = _feat_matrix(reviews)
    for j, c in enumerate(FEAT_COLS):
        df[c] = M[:, j]
    res = [{} for _ in reviews]
    for n, pipe in sk.items():
        try:
            proba = pipe.predict_proba(df)[:, 1]
            t = thr[n]
            vec = pipe.named_steps["feat"].named_transformers_["tfidf"]
            fn = vec.get_feature_names_out()
            X = vec.transform(texts)
            for i in range(len(reviews)):
                label = int(proba[i] >= t)
                row = X[i].toarray().ravel()
                top = [fn[k] for k in row.argsort()[::-1][:5] if row[k] > 0]
                reasons = [f'"{w}"' for w in top] or ["overall text + behavioral pattern"]
                res[i][n] = {"label": label, "confidence": _conf(proba[i], label), "reasons": reasons}
        except Exception as e:
            for r in res:
                r[n] = {"label": None, "confidence": None, "reasons": [f"error: {e}"]}
    return res


def _score_sklearn(reviews):
    """Cache-aware sklearn scoring (keyed by review_id). Reuses results across variants and
    incremental loads — sklearn is recomputed ONLY for reviews not already in _sk_review_cache.
    Reviews without a review_id fall through to a fresh compute (not cached)."""
    if not reviews:
        return []
    results = [None] * len(reviews)
    todo = []
    for i, r in enumerate(reviews):
        rid = r.get("review_id")
        hit = _sk_review_cache.get(rid) if rid is not None else None
        if hit is not None:
            results[i] = hit
        else:
            todo.append(i)
    if todo:
        computed = _compute_sklearn([reviews[i] for i in todo])
        for k, i in enumerate(todo):
            results[i] = computed[k]
            rid = reviews[i].get("review_id")
            if rid is not None and computed[k]:          # don't cache empty (sklearn-unavailable) results
                _sk_review_cache[rid] = computed[k]
        if len(_sk_review_cache) > _SK_CACHE_MAX:         # bound: drop the oldest-inserted overflow
            for key in list(_sk_review_cache)[:len(_sk_review_cache) - _SK_CACHE_MAX]:
                _sk_review_cache.pop(key, None)
    return results


def _lstm_probs(texts, beh_raw, variant="standard"):
    """Raw fake probabilities from the multimodal LSTM for parallel texts/features."""
    sess, tok, scaler, _ = _load_lstm(variant)
    textarr = _pad(tok.texts_to_sequences(texts), LSTM_MAX_LEN)
    beh = scaler.transform(beh_raw).astype(np.float32)
    return sess.run(None, {"text_input": textarr, "beh_input": beh})[0].reshape(-1)


def _db_probs(texts, beh_raw, variant="standard"):
    """Raw fake probabilities from the multimodal DistilBERT (sigmoid of the logit)."""
    sess, tok, scaler, _ = _load_db(variant)
    beh_all = scaler.transform(beh_raw).astype(np.float32)
    out = []
    B = 16
    for s in range(0, len(texts), B):
        enc = tok(texts[s:s + B], truncation=True, padding="max_length",
                  max_length=DB_MAX_LEN, return_tensors="np")
        feed = {"input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
                "beh": beh_all[s:s + B]}
        logit = np.asarray(sess.run(None, feed)[0], dtype=np.float64).reshape(-1)
        out.append(1.0 / (1.0 + np.exp(-logit)))
    return np.concatenate(out) if out else np.array([])


def _score_lstm(reviews, variant="standard"):
    if not _load_lstm(variant):
        return _none(reviews, "LSTM unavailable")
    thr = _load_lstm(variant)[3]
    probs = _lstm_probs(_texts(reviews), _feat_matrix(reviews), variant)
    out = []
    for p in probs:
        label = int(p >= thr)
        out.append({"label": label, "confidence": _conf(p, label),
                    "reasons": ["text + reviewer/business behavioral signal (multi-modal BiLSTM)"]})
    return out


def _score_db(reviews, variant="standard"):
    if not _load_db(variant):
        return _none(reviews, "DistilBERT unavailable")
    thr = _load_db(variant)[3]
    probs = _db_probs(_texts(reviews), _feat_matrix(reviews), variant)
    out = []
    for p in probs:
        label = int(p >= thr)
        out.append({"label": label, "confidence": _conf(p, label),
                    "reasons": ["text + reviewer/business behavioral signal (multi-modal DistilBERT)"]})
    return out


def _dl_sentences(review, variant="standard"):
    """Per-sentence LSTM/DistilBERT verdicts (the review's behavioral vector held
    constant across sentences), so DL dots can sentence-highlight."""
    sents = _split_sentences(str(review.get("review_text", "") or ""))
    if not sents:
        return [], []
    susp, genu = {}, {}
    beh1 = _feat_matrix([review])
    beh_n = np.repeat(beh1, len(sents), axis=0)
    if _load_lstm(variant):
        try:
            thr = _load_lstm(variant)[3]
            for i, p in enumerate(_lstm_probs(sents, beh_n, variant)):
                (susp if p >= thr else genu).setdefault(sents[i], set()).add("LSTM")
        except Exception:
            pass
    if _load_db(variant):
        try:
            thr = _load_db(variant)[3]
            for i, p in enumerate(_db_probs(sents, beh_n, variant)):
                (susp if p >= thr else genu).setdefault(sents[i], set()).add("DistilBERT")
        except Exception:
            pass
    susp_l = [{"sentence": s, "models": sorted(m)} for s, m in susp.items()]
    genu_l = [{"sentence": s, "models": sorted(m), "type": "genuine"} for s, m in genu.items()]
    return susp_l, genu_l


def score_yelp_fake(reviews, variant="standard"):
    """Per-review bundle for the Yelp Fake Review Detection section:
        {fake_models, suspicious_phrases, genuine_phrases,
         dl_suspicious_sentences, dl_genuine_sentences}
    fake_models = 8 multimodal sklearn (SHARED across variants) + the variant's multimodal
    lstm + distilbert. variant ∈ {standard, focal, contrastive} swaps only the DL members.
    Each review dict must carry the behavioral feature columns (FEAT_COLS)."""
    if not reviews:
        return []
    sk = _score_sklearn(reviews)
    ls = _score_lstm(reviews, variant)
    db = _score_db(reviews, variant)
    out = []
    for i, r in enumerate(reviews):
        fm = dict(sk[i]); fm["lstm"] = ls[i]; fm["distilbert"] = db[i]
        text = str(r.get("review_text", "") or "")
        dl_s, dl_g = _dl_sentences(r, variant)
        out.append({
            "fake_models":             fm,
            "suspicious_phrases":      _get_suspicious_phrases(text, fm),
            "genuine_phrases":         _get_genuine_phrases(text, fm),
            "dl_suspicious_sentences": dl_s,
            "dl_genuine_sentences":    dl_g,
        })
    return out


def score_yelp_distilbert(reviews, variant="standard"):
    """DistilBERT-only multimodal verdicts (for the Yelp user-history chart).
    Returns [{label, confidence, reasons}] per review."""
    if not reviews:
        return []
    return _score_db(reviews, variant)

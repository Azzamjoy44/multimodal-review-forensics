# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
download_negatives.py — build the NON-REVIEW (negative) class for the
review / non-review classifier.

Pulls ~340k diverse non-review English texts from HuggingFace + two academic
sources (CMU Movie Summaries, McAuley Google Local) — no Kaggle auth needed.
Writes data/non_reviews.csv with columns:
    text, source, label   (label is always 0 = non-review)

Composition (see THESIS_NOTES / CLAUDE.md design discussion). Several sources
are MATCHED negatives — same topic as a review domain, but descriptive not
evaluative:
    Amazon product meta  (product descriptions)   ~45k  HARD  <-> amazon reviews
    Amazon Books meta    (book descriptions)       ~30k  HARD  <-> book reviews
    Amazon Software meta (app/software desc)       ~30k  HARD  <-> app reviews
    Google Local meta    (business descriptions)   ~30k  HARD  <-> yelp reviews
    CMU Movie Summaries  (movie plots)             ~30k  HARD  <-> movie reviews
    FronkonGames Steam   (game store copy)         ~30k  HARD  <-> game reviews
    Amazon-QA  query     (product questions)       ~40k  MEDIUM
    SQuAD      question   (general questions)       ~25k  MEDIUM
    AG News    text       (news)                    ~40k  EASY
    WikiText-103 text     (encyclopedic)            ~40k  EASY
Plus broad GENRE coverage (non-review text types beyond our review domains, so
the gate rejects anything that isn't a review - not just familiar genres):
    corbt/all-recipes        (how-to / instructions)  ~6k
    empathetic_dialogues     (casual conversation)    ~6k
    aeslc                    (email correspondence)   ~6k
    lex_glue/eurlex          (legal / regulatory)     ~6k
    code_search_net          (source code)            ~6k
    writingprompts           (fiction / narrative)    ~6k
    ML-ArXiv-Papers          (academic abstracts)     ~6k
    merve/poetry             (poetry / verse)         ~0.5k
Total ~380k, downsampled to balance the 6-domain (~300k) review positive class.
Every review domain has a matched (same-topic, descriptive) hard negative, and
the broad genre block closes the "non-review genre we never trained on" gap.
"""

import os
import re
import sys
import csv
import json
import gzip
import html as _html
import tarfile
import urllib.request

from datasets import load_dataset
from huggingface_hub import hf_hub_url

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_CSV = os.path.join(DATA_DIR, "non_reviews.csv")

MIN_WORDS = 5         # drop degenerate one-/two-word fragments
MAX_CHARS = 5000      # cap absurdly long encyclopedic dumps

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
_ws  = re.compile(r"\s+")
_tag = re.compile(r"<[a-zA-Z/!][^>]*>")
_url = re.compile(r"https?://\S+|www\.\S+")


def clean(text):
    """Strip HTML markup/entities + URLs, normalise whitespace; return None if
    too short/empty.

    HTML tags (<br/>, <a ...>, ...) appear almost exclusively in source REVIEW
    text (IMDB/Amazon/Books), so leaving them in would be a spurious 'this is a
    review' shortcut that wrecks real-world generalisation. URLs are similarly
    skewed (app descriptions), so they are stripped too."""
    if text is None:
        return None
    text = _html.unescape(str(text))   # &amp; -> &, &lt;br&gt; -> <br>  (so tags match)
    text = _tag.sub(" ", text)         # remove HTML tags
    text = _url.sub(" ", text)         # remove URLs
    text = _ws.sub(" ", text).strip()
    if not text:
        return None
    if len(text.split()) < MIN_WORDS:
        return None
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0]
    return text


def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# per-source collectors  (each yields cleaned strings)
# ---------------------------------------------------------------------------
def _stream_meta_category(cat, target):
    """Yield cleaned 'title + features + description' strings from one Amazon
    Reviews 2023 metadata category, streaming the raw meta jsonl over HTTP
    line-by-line (datasets 4.x refuses the script-based loader)."""
    url = hf_hub_url(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename=f"raw/meta_categories/meta_{cat}.jsonl",
        repo_type="dataset",
    )
    got = 0
    req = urllib.request.Request(url, headers={"User-Agent": "thesis-downloader"})
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            try:
                row = json.loads(raw)
            except Exception:
                continue
            parts = []
            if row.get("title"):
                parts.append(str(row["title"]))
            feats = row.get("features") or []
            if isinstance(feats, list):
                parts.extend(str(f) for f in feats)
            desc = row.get("description") or []
            if isinstance(desc, list):
                parts.extend(str(d) for d in desc)
            elif desc:
                parts.append(str(desc))
            t = clean(" ".join(parts))
            if t:
                yield t
                got += 1
                if got >= target:
                    break
    return got


def collect_amazon_meta(target):
    """Product descriptions from several Amazon product categories (matched
    negative for amazon product reviews) for vocabulary diversity. Books and
    Software are split into their own dedicated sources below."""
    cats = [
        "Electronics",
        "Home_and_Kitchen",
        "Clothing_Shoes_and_Jewelry",
        "Sports_and_Outdoors",
        "Office_Products",
    ]
    per_cat = target // len(cats) + 1
    for cat in cats:
        n = 0
        for t in _stream_meta_category(cat, per_cat):
            yield t
            n += 1
        log(f"  amazon_meta[{cat}]: {n}")


def collect_book_desc(target):
    """Book descriptions / publisher blurbs (Amazon Books metadata) — the
    matched negative for book reviews: describes the book without evaluating it
    (the text most easily confused with a plot-summarising book review)."""
    n = 0
    for t in _stream_meta_category("Books", target):
        yield t
        n += 1
    log(f"  book_desc[Books]: {n}")


def collect_app_software_desc(target):
    """Software/app store descriptions (Amazon Software metadata) — the matched
    negative for app reviews: same topic (apps/software), purely descriptive."""
    n = 0
    for t in _stream_meta_category("Software", target):
        yield t
        n += 1
    log(f"  app_software_desc[Software]: {n}")


def collect_business_desc(target):
    """Local-business descriptions (McAuley Google Local place metadata) — the
    matched negative for Yelp reviews: factually describes a restaurant / shop /
    service without evaluating it. Streamed + gunzipped over HTTP.

    Google Local has many chain locations sharing identical boilerplate
    descriptions, so we dedup locally and sweep several states to reach the
    target count of *unique* descriptions."""
    states = ["California", "Texas", "New_York", "Florida",
              "Illinois", "Pennsylvania", "Ohio", "Georgia"]
    base = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/meta-{}.json.gz"
    seen_local = set()
    got = 0
    for state in states:
        req = urllib.request.Request(base.format(state),
                                     headers={"User-Agent": "thesis-downloader"})
        try:
            resp = urllib.request.urlopen(req)
        except Exception:
            continue
        with resp:
            gz = gzip.GzipFile(fileobj=resp)
            for raw in gz:
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                t = clean(row.get("description"))
                if not t:
                    continue
                k = t.lower()
                if k in seen_local:
                    continue
                seen_local.add(k)
                yield t
                got += 1
                if got >= target:
                    break
        log(f"  business_desc[{state}]: running total {got}")
        if got >= target:
            break


def collect_cmu_movies(target):
    """CMU Movie Summary Corpus plot summaries (academic wget)."""
    url = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    tgz = os.path.join(DATA_DIR, "MovieSummaries.tar.gz")
    if not os.path.exists(tgz):
        log("  downloading CMU MovieSummaries.tar.gz ...")
        urllib.request.urlretrieve(url, tgz)
    got = 0
    with tarfile.open(tgz, "r:gz") as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("plot_summaries.txt"))
        fh = tf.extractfile(member)
        for line in fh:
            line = line.decode("utf-8", "ignore")
            # format: wiki_movie_id <TAB> summary
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            t = clean(parts[1])
            if t:
                yield t
                got += 1
                if got >= target:
                    break
    log(f"  cmu_movies: {got}")


def collect_steam(target):
    """FronkonGames Steam store descriptions (CC BY 4.0)."""
    got = 0
    ds = load_dataset("FronkonGames/steam-games-dataset", split="train", streaming=True)
    for row in ds:
        t = clean(
            row.get("detailed_description")
            or row.get("about_the_game")
            or row.get("short_description")
        )
        if t:
            yield t
            got += 1
            if got >= target:
                break
    log(f"  steam: {got}")


def collect_amazon_qa(target):
    """Product-topic questions from embedding-data/Amazon-QA (field: query)."""
    got = 0
    ds = load_dataset("embedding-data/Amazon-QA", split="train", streaming=True)
    for row in ds:
        q = row.get("query")
        if isinstance(q, list):
            q = q[0] if q else None
        t = clean(q)
        if t:
            yield t
            got += 1
            if got >= target:
                break
    log(f"  amazon_qa: {got}")


def collect_squad(target):
    """General-knowledge questions from SQuAD (field: question)."""
    got = 0
    ds = load_dataset("rajpurkar/squad", split="train")
    for row in ds:
        t = clean(row.get("question"))
        if t:
            yield t
            got += 1
            if got >= target:
                break
    log(f"  squad: {got}")


def collect_agnews(target):
    """News headlines + descriptions from AG News (field: text)."""
    got = 0
    ds = load_dataset("fancyzhx/ag_news", split="train")
    for row in ds:
        t = clean(row.get("text"))
        if t:
            yield t
            got += 1
            if got >= target:
                break
    log(f"  ag_news: {got}")


def collect_wikitext(target):
    """Encyclopedic prose from WikiText-103 (filter blank + heading lines)."""
    got = 0
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                      split="train", streaming=True)
    for row in ds:
        line = row.get("text") or ""
        s = line.strip()
        if not s or s.startswith("="):   # skip blanks and "= Heading ="
            continue
        t = clean(line)
        if t:
            yield t
            got += 1
            if got >= target:
                break
    log(f"  wikitext: {got}")


# ---------------------------------------------------------------------------
# Broad non-review GENRE coverage (NOT topic-matched) - so the gate rejects any
# kind of non-review text a user might paste, not just the genres of our review
# sources. (The recipe miss in the generalisation test exposed this gap.)
# ---------------------------------------------------------------------------
def _hf_stream_field(repo, field, target, config=None):
    """Yield up to `target` cleaned strings from one text field of a streamed
    HuggingFace dataset."""
    ds = (load_dataset(repo, config, split="train", streaming=True) if config
          else load_dataset(repo, split="train", streaming=True))
    n = 0
    for row in ds:
        t = clean(row.get(field))
        if t:
            yield t
            n += 1
            if n >= target:
                break
    log(f"  {repo}: {n}")


def collect_how_to(target):
    """Recipes / instructional how-to text (imperative steps)."""
    yield from _hf_stream_field("corbt/all-recipes", "input", target)


def collect_code(target):
    """Source code - function definitions with docstrings."""
    yield from _hf_stream_field("code_search_net", "whole_func_string", target, config="python")


def collect_legal(target):
    """Legal / regulatory prose (EU legislation)."""
    yield from _hf_stream_field("coastalcph/lex_glue", "text", target, config="eurlex")


def collect_email(target):
    """Email correspondence (Enron / AESLC)."""
    yield from _hf_stream_field("aeslc", "email_body", target)


def collect_fiction(target):
    """Creative fiction / narrative prose (WritingPrompts stories)."""
    yield from _hf_stream_field("euclaise/writingprompts", "story", target)


def collect_poetry(target):
    """Poetry / verse."""
    yield from _hf_stream_field("merve/poetry", "content", target)


def collect_academic(target):
    """Academic / scientific abstracts (arXiv ML papers)."""
    yield from _hf_stream_field("CShorten/ML-ArXiv-Papers", "abstract", target)


def collect_dialogue(target):
    """Casual conversational utterances (empathetic dialogues) - chit-chat, not
    product/service evaluations."""
    ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train", streaming=True)
    n = 0
    for row in ds:
        for turn in (row.get("conversations") or []):
            c = turn.get("content") if isinstance(turn, dict) else turn
            t = clean(c)
            if t:
                yield t
                n += 1
                if n >= target:
                    log(f"  dialogue: {n}")
                    return
    log(f"  dialogue: {n}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
SOURCES = [
    ("amazon_meta",       collect_amazon_meta,       45_000),
    ("book_desc",         collect_book_desc,         30_000),
    ("app_software_desc", collect_app_software_desc, 30_000),
    ("business_desc",     collect_business_desc,     28_000),
    ("cmu_movies",        collect_cmu_movies,        30_000),
    ("steam_desc",        collect_steam,             30_000),
    ("amazon_qa",         collect_amazon_qa,         40_000),
    ("squad",             collect_squad,             25_000),
    ("ag_news",           collect_agnews,            40_000),
    ("wikitext",          collect_wikitext,          40_000),
    # broad genre coverage (non-review text types beyond our review domains)
    ("how_to",            collect_how_to,             6_000),
    ("dialogue",          collect_dialogue,           6_000),
    ("email",             collect_email,              6_000),
    ("legal",             collect_legal,              6_000),
    ("code",              collect_code,               6_000),
    ("fiction",           collect_fiction,            6_000),
    ("academic",          collect_academic,           6_000),
    ("poetry",            collect_poetry,             6_000),
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    seen = set()
    rows = []  # (text, source)
    for name, fn, target in SOURCES:
        log(f"[{name}] target {target} ...")
        kept = 0
        for t in fn(target):
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append((t, name))
            kept += 1
        log(f"[{name}] kept {kept} (after global dedup)")

    log(f"TOTAL non-review rows: {len(rows)}")
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "source", "label"])
        for text, source in rows:
            w.writerow([text, source, 0])
    log(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()

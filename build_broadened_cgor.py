"""
build_broadened_cgor.py — assemble a BROADENED, MULTI-DOMAIN CG/OR
(machine-generated vs human) fake-review dataset. Leaves the original
`data/fake reviews dataset.csv` untouched.

Label:  1 = CG (machine-generated / fake)   0 = OR (original / human)

This is the training set for the GENERAL all-domain AI-fake detector ("Model A"),
the core of the Dashboard catch-all. It covers SIX domains so the 8 sklearn + LSTM
+ DistilBERT models learn machine-vs-human across review types, not just products:

    products · movies · restaurants · apps · books · games

Composition — class-balanced AND length-balanced per (domain, word-length bin), so
neither domain nor length is a shortcut:

  CG (machine-generated), tagged by generator family (for leave-one-generator-out):
    products    : Salminen CG · sutro · Kenshiii · openai_gpt · open-models(qwen/llama)
    movies      : Lyra(AI) · domains-AI(movies)
    restaurants : domains-AI(restaurants) · ai_restaurant_reviews_open(qwen/llama/zephyr/yi)
    apps/books/games : domains-AI(<domain>)
    (domains-AI = ai_generated_reviews_domains_open.csv [qwen/llama/zephyr] +
                  ai_generated_reviews_domains.csv [gpt])
  OR (human), per domain:
    products    : real Amazon (Reviews.csv) · Salminen OR · review_detection(amazon)
    movies      : real IMDB · Lyra(human) · review_detection(imdb)
    restaurants : review_detection(yelp)
    apps        : review_detection(apps)
    books       : review_detection(books)
    games       : review_detection(steam)
  (review_detection.csv label==1 rows are real human reviews, pre-cleaned, tagged by source.)

Per-domain CG is capped (MAX_CG_PER_DOMAIN) so the data-rich products domain can't
swamp the others — important for cross-domain generalization.

Output: data/fake_reviews_broadened.csv  (text, label, source, domain, split)
        (optional CLI arg = alternate output path, for previewing without clobbering)

RUN
    python build_broadened_cgor.py
    python build_broadened_cgor.py data/_preview.csv   # write elsewhere (validation)
"""

import os
import sys
import csv
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from download_negatives import clean, DATA_DIR

ORIG_CGOR = os.path.join(DATA_DIR, "fake reviews dataset.csv")
AMAZON    = os.path.join(DATA_DIR, "Reviews.csv")
IMDB      = os.path.join(DATA_DIR, "IMDB Dataset.csv")
REVIEW_DET = os.path.join(DATA_DIR, "review_detection.csv")   # local, pre-cleaned, per-domain human reviews

# AI (CG) sources
AI_GEN_PROD   = os.path.join(DATA_DIR, "ai_generated_reviews.csv")            # products, openai
AI_OPEN_PROD  = os.path.join(DATA_DIR, "ai_generated_reviews_open.csv")        # products, qwen/llama
AI_DOMAINS_OPEN = os.path.join(DATA_DIR, "ai_generated_reviews_domains_open.csv")  # 5 domains, qwen/llama/zephyr
AI_DOMAINS_GPT  = os.path.join(DATA_DIR, "ai_generated_reviews_domains.csv")       # 5 domains, gpt
AI_REST_OPEN    = os.path.join(DATA_DIR, "ai_restaurant_reviews_open.csv")         # restaurants, qwen/llama/zephyr/yi (scale-up)
AI_SHORT_OPEN   = os.path.join(DATA_DIR, "ai_short_reviews_open.csv")               # SHORT length-matched, 5 domains (fills short bins for apps/games/books/...)
AI_OLDGEN_OPEN  = os.path.join(DATA_DIR, "ai_oldgen_reviews_open.csv")               # OLD/weak family (distilgpt2/gpt2/opt) across all 6 domains -> fixes salminen/products LOGO/LOD gap

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "fake_reviews_broadened.csv")
SUTRO_CAP = 20_000
MAX_CG_PER_DOMAIN = 20_000   # cap so products can't swamp the smaller domains
SEED = 42
random.seed(SEED)

# review_detection.csv source -> our domain (human OR)
RD_SRC_TO_DOMAIN = {"amazon": "products", "imdb": "movies", "yelp": "restaurants",
                    "books": "books", "apps": "apps", "steam": "games"}
DOMAINS = ["products", "movies", "restaurants", "apps", "books", "games"]


def log(m):
    print(m, flush=True)


def main():
    seen = set()
    cg_rows = []   # (text, 1, source, domain)
    or_rows = []   # (text, 0, source, domain)

    def add_cg(text, source, domain):
        t = clean(text)
        if not t:
            return False
        k = t.lower()
        if k in seen:
            return False
        seen.add(k)
        cg_rows.append((t, 1, source, domain))
        return True

    def add_or(text, source, domain):
        t = clean(text)
        if not t:
            return
        k = t.lower()
        if k in seen:
            return
        seen.add(k)
        or_rows.append((t, 0, source, domain))

    def ingest_ai_csv(path, label, default_src, src_col=None):
        """Add CG rows from a domains-AI csv (has a 'domain' column)."""
        if not os.path.exists(path):
            log(f"  {os.path.basename(path)} missing, skipping"); return
        df = pd.read_csv(path)
        dom_col = "domain" if "domain" in df.columns else None
        n = 0
        for _, r in df.iterrows():
            dom = r[dom_col] if dom_col else "products"
            if dom not in DOMAINS:
                continue
            src = str(r[src_col]) if (src_col and src_col in df.columns) else default_src
            if src.startswith("gpt-"):   # collapse OpenAI gpt-4o-mini/gpt-3.5-turbo -> "gpt" (NOT gpt2/gpt2med)
                src = "gpt"
            if add_cg(str(r["text"]), src, dom):
                n += 1
        log(f"  {label}: +{n}")

    # ------------------------------------------------------------ CG (machine) --
    log("CG (machine-generated):")
    df0 = pd.read_csv(ORIG_CGOR)
    n = sum(add_cg(t, "salminen_cg", "products") for t in df0[df0["label"].str.upper() == "CG"]["text_"])
    log(f"  salminen_cg (products): {n}")

    from datasets import load_dataset
    n = 0
    for r in load_dataset("sutro/synthetic-product-reviews-20k", split="train", streaming=True):
        if add_cg(r.get("review_text"), "sutro", "products"):
            n += 1
        if n >= SUTRO_CAP:
            break
    log(f"  sutro (products): {n}")
    n = sum(add_cg(r.get("review_text"), "kenshiii", "products")
            for r in load_dataset("Kenshiii/synthetic-product-reviews", split="train", streaming=True))
    log(f"  kenshiii (products): {n}")

    if os.path.exists(AI_GEN_PROD):
        n = sum(add_cg(t, "openai_gpt", "products") for t in pd.read_csv(AI_GEN_PROD)["text"])
        log(f"  openai_gpt (products): {n}")
    if os.path.exists(AI_OPEN_PROD):
        dfo = pd.read_csv(AI_OPEN_PROD)
        n = sum(add_cg(t, str(s), "products") for t, s in zip(dfo["text"], dfo["source"]))
        log(f"  open-models (products): {n}")

    # Lyra AI movie reviews (label==1); stash human (label==0) for movie OR
    lyra_human = []
    n = 0
    for r in load_dataset("Lyra-stellAI/AI_Human_generated_movie_reviews", split="train", streaming=True):
        if r.get("labels") == 1:
            if add_cg(r.get("text"), "lyra", "movies"):
                n += 1
        elif r.get("labels") == 0:
            lyra_human.append(r.get("text"))
    log(f"  lyra (movies): {n}  | lyra human stashed: {len(lyra_human)}")

    # multi-domain AI (movies/restaurants/apps/books/games)
    ingest_ai_csv(AI_DOMAINS_OPEN, "domains-AI open (qwen/llama/zephyr)", "open", src_col="source")
    ingest_ai_csv(AI_DOMAINS_GPT,  "domains-AI gpt",                       "gpt", src_col="model")
    ingest_ai_csv(AI_REST_OPEN,    "restaurant scale-up (qwen/llama/zephyr/yi)", "open", src_col="source")
    ingest_ai_csv(AI_SHORT_OPEN,   "short length-matched (qwen/llama/zephyr)",   "open", src_col="source")
    ingest_ai_csv(AI_OLDGEN_OPEN,  "old/weak family (distilgpt2/gpt2/opt)",      "open", src_col="source")

    cg_df = pd.DataFrame(cg_rows, columns=["text", "label", "source", "domain"])
    log("\nCG per domain (pre-cap):\n" + cg_df["domain"].value_counts().to_string())

    # cap each domain's CG so products can't dominate
    capped = []
    for dom, g in cg_df.groupby("domain"):
        if len(g) > MAX_CG_PER_DOMAIN:
            g = g.sample(n=MAX_CG_PER_DOMAIN, random_state=SEED)
        capped.append(g)
    cg_df = pd.concat(capped).reset_index(drop=True)
    log(f"CG total after cap ({MAX_CG_PER_DOMAIN}/domain): {len(cg_df)}")

    # ------------------------------------------------------------ OR (human) --
    log("\nOR (human) sources:")
    for t in df0[df0["label"].str.upper() == "OR"]["text_"]:
        add_or(t, "salminen_or", "products")
    for t in pd.read_csv(AMAZON, usecols=["Text"]).sample(frac=1, random_state=SEED)["Text"]:
        add_or(t, "amazon_real", "products")
    for t in lyra_human:
        add_or(t, "lyra_human", "movies")
    for t in pd.read_csv(IMDB, usecols=["review"]).sample(frac=1, random_state=SEED)["review"]:
        add_or(t, "imdb_real", "movies")
    # per-domain human reviews from review_detection.csv (label==1 = real reviews)
    if os.path.exists(REVIEW_DET):
        rd = pd.read_csv(REVIEW_DET, usecols=["text", "label", "source"])
        rd = rd[rd["label"] == 1]
        for src, dom in RD_SRC_TO_DOMAIN.items():
            for t in rd[rd["source"] == src]["text"]:
                add_or(t, f"rd_{src}", dom)
    or_df = pd.DataFrame(or_rows, columns=["text", "label", "source", "domain"])
    log("OR per domain pool:\n" + or_df["domain"].value_counts().to_string())

    # ---- domain + length matched: per (domain, bin) keep min(CG, OR) of each ----
    BINS = [0, 20, 40, 60, 80, 100, 130, 160, 200, 250, 300, 400, 10**9]
    cg_df["_bin"] = pd.cut(cg_df["text"].str.split().str.len(), BINS, labels=False)
    or_df["_bin"] = pd.cut(or_df["text"].str.split().str.len(), BINS, labels=False)

    keep, cols = [], ["text", "label", "source", "domain"]
    for (dom, b), cgg in cg_df.groupby(["domain", "_bin"]):
        org = or_df[(or_df.domain == dom) & (or_df["_bin"] == b)]
        k = min(len(cgg), len(org))
        if k:
            keep.append(cgg[cols].sample(n=k, random_state=SEED))
            keep.append(org[cols].sample(n=k, random_state=SEED))
    df = pd.concat(keep).reset_index(drop=True)

    log(f"\nTOTAL {len(df)} | CG {int((df.label==1).sum())} | OR {int((df.label==0).sum())}")
    log("by domain x label:\n" + pd.crosstab(df.domain, df.label).to_string())
    log("CG by source:\n" + df[df.label == 1].source.value_counts().to_string())

    idx = df.index.to_numpy()
    tr, tmp = train_test_split(idx, test_size=0.20, random_state=SEED, stratify=df["label"])
    va, te = train_test_split(tmp, test_size=0.50, random_state=SEED, stratify=df.loc[tmp, "label"])
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    log("\nsplit x label:\n" + pd.crosstab(df.split, df.label).to_string())

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)
    log(f"\nwrote {OUT}  ({len(df)} rows)")


if __name__ == "__main__":
    main()

"""
build_yelp_multimodal_augmented.py
----------------------------------
Phase 2 of the multi-modal Yelp detector: turn it into a GENERALIZED / catch-all
fake detector that flags BOTH
  * human-written behavioural fakes (Yelp spam — caught by the behavioural branch), and
  * AI-generated reviews            (caught by the text branch).

It takes the Yelp feature table (data/yelp_multimodal_features.csv) and injects
AI-generated RESTAURANT reviews as additional fakes (label 1). Restaurant domain
is chosen on purpose: the genuine Yelp reviews are also restaurant text, so the
only thing separating an AI fake from a genuine review is WRITING STYLE, not topic.

Key design choice — AI reviews are given a behavioural profile COPIED from a real
*genuine* Yelp reviewer (only the text-length features are overwritten with the AI
review's own length). So an AI fake looks behaviourally innocent; the model can
only catch it from the TEXT. This forces the text branch to learn the AI signal
instead of cheating on a behavioural tell.

Each row is tagged `fake_type` in {genuine, human_fake, ai_fake} so the evaluator
can report human-fake recall, AI-fake recall and genuine specificity separately.

Output: data/yelp_multimodal_augmented.csv

RUN
    python build_yelp_multimodal_augmented.py
"""

import os
import csv

import numpy as np
import pandas as pd

from download_negatives import clean

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
FEATS     = os.path.join(DATA_DIR, "yelp_multimodal_features.csv")
AI_REST_OPEN = os.path.join(DATA_DIR, "ai_restaurant_reviews_open.csv")        # NEW: multi-generator open (qwen/llama3/zephyr/yi)
AI_OPEN_OLD  = os.path.join(DATA_DIR, "ai_generated_reviews_domains_open.csv")  # fallback if the new file is absent
AI_API       = os.path.join(DATA_DIR, "ai_generated_reviews_domains.csv")       # existing GPT reviews (already generated; no new spend)
OUT       = os.path.join(DATA_DIR, "yelp_multimodal_augmented.csv")
SEED      = 42
AI_SPLIT  = (0.8, 0.1, 0.1)   # train / val / test for the injected AI fakes
AI_TARGET = 12_000            # max AI fakes to keep after length-stratification
LEN_BINS  = [0, 20, 40, 60, 80, 100, 130, 160, 200, 250, 300, 400, 10**9]


def log(m):
    print(m, flush=True)


def main():
    rng = np.random.RandomState(SEED)
    df = pd.read_csv(FEATS)
    df["text"] = df["text"].astype(str)
    feat_cols = [c for c in df.columns if c not in ("text", "label", "split", "source")]
    df["fake_type"] = np.where(df["label"] == 1, "human_fake", "genuine")
    df["generator"] = ""   # only meaningful for ai_fake rows (used by leave-one-generator-out eval)
    log(f"Yelp rows: {len(df):,}  (human_fake={int((df.label==1).sum()):,}, "
        f"genuine={int((df.label==0).sum()):,})  features={len(feat_cols)}")

    # ---- collect AI restaurant reviews + their generator, clean + dedupe -- #
    # Each AI review keeps its generator family (qwen/llama3/zephyr/yi/gpt) so we can
    # run leave-one-generator-out (does the model catch reviews from an UNSEEN generator?).
    seen = set(df["text"].str.lower())
    ai_items = []   # (text, generator)

    def ingest(path, get_gen, label):
        if not os.path.exists(path):
            log(f"  {os.path.basename(path)} missing, skipping"); return
        a = pd.read_csv(path)
        if "domain" in a.columns:
            a = a[a["domain"] == "restaurants"]
        n = 0
        for _, row in a.iterrows():
            t = clean(str(row["text"]))
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            ai_items.append((t, get_gen(row)))
            n += 1
        log(f"  AI restaurant ({label}): +{n}")

    # open models: prefer the new multi-generator file; else fall back to the old 1.5k
    if os.path.exists(AI_REST_OPEN):
        ingest(AI_REST_OPEN, lambda r: str(r.get("source", "open")), "open multi-gen")
    else:
        ingest(AI_OPEN_OLD, lambda r: str(r.get("source", "open")), "open (old fallback)")
    # GPT: reuse the already-generated reviews (no new API spend) as a distinct family
    ingest(AI_API, lambda r: "gpt", "gpt (existing)")

    log(f"total unique AI restaurant fakes (pre-stratify): {len(ai_items):,}  "
        f"per generator: {pd.Series([g for _, g in ai_items]).value_counts().to_dict()}")

    # ---- LENGTH-STRATIFY the AI pool to match the real Yelp length distribution.
    # AI reviews skew long; if left unmatched, review length alone separates AI from
    # real reviews (a shortcut). Sampling AI per word-length bin in the same
    # proportions as the real Yelp reviews makes length carry ~no signal (AUC ~0.5),
    # so the model must catch AI on writing STYLE, not length. ----------------- #
    yelp_bins = pd.cut(df["review_word_len"], LEN_BINS, labels=False)
    yelp_share = yelp_bins.value_counts(normalize=True)
    ai_bin = pd.cut([len(t.split()) for t, _ in ai_items], LEN_BINS, labels=False)
    from collections import defaultdict
    bin_idx = defaultdict(list)
    for i, b in enumerate(ai_bin):
        bin_idx[b].append(i)
    target = min(len(ai_items), AI_TARGET)
    keep = []
    for b, share in yelp_share.items():
        want = int(round(share * target))
        avail = bin_idx.get(b, [])
        take = min(want, len(avail))
        if take:
            keep += list(rng.choice(avail, size=take, replace=False))
    ai_items = [ai_items[i] for i in keep]
    _ai_wl = [len(t.split()) for t, _ in ai_items]
    log(f"after length-stratify: {len(ai_items):,} AI fakes | median words "
        f"{int(pd.Series(_ai_wl).median()) if _ai_wl else 0} (target ~ real Yelp median)")

    # ---- behavioural donors: real GENUINE reviewers (per AI review) ------- #
    genuine = df[df["label"] == 0].reset_index(drop=True)
    donor_idx = rng.randint(0, len(genuine), size=len(ai_items))

    # split assignment for AI rows
    r = rng.rand(len(ai_items))
    splits = np.where(r < AI_SPLIT[0], "train",
             np.where(r < AI_SPLIT[0] + AI_SPLIT[1], "val", "test"))

    ai_rows = []
    for i, (txt, gen) in enumerate(ai_items):
        feats = genuine.loc[donor_idx[i], feat_cols].to_dict()   # behaviourally innocent
        feats["review_word_len"] = len(txt.split())              # ...but real text length
        feats["review_char_len"] = len(txt)
        row = {"text": txt, "label": 1, "split": splits[i],
               "source": "ai_restaurant", "fake_type": "ai_fake", "generator": gen}
        row.update(feats)
        ai_rows.append(row)
    ai_df = pd.DataFrame(ai_rows)[df.columns]   # column order match (includes 'generator')

    out = pd.concat([df, ai_df], ignore_index=True)
    out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)
    out.to_csv(OUT, index=False, quoting=csv.QUOTE_MINIMAL)

    log("\naugmented dataset:")
    log("by split x fake_type:\n" + pd.crosstab(out["split"], out["fake_type"]).to_string())
    ai = out[out["fake_type"] == "ai_fake"]
    log("\nAI fakes by split x generator:\n" + pd.crosstab(ai["split"], ai["generator"]).to_string())
    log(f"\nwrote {OUT}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()

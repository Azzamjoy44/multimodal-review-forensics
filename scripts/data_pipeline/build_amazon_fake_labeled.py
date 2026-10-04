# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
build_amazon_fake_labeled.py — labeled Amazon fake-review browse set for the Fake Review Detection
section, so the UI can show predicted-vs-ground-truth (like Yelp). Uses the HELD-OUT test split of
the combined corpus (products domain only): real Amazon reviews (genuine) + AI-generated product
reviews (cg_fake). The unified detector was trained on this corpus but NOT on the test split, so
this is an honest in-distribution check. Output: data/amazon_fake_labeled.csv
RUN: python build_amazon_fake_labeled.py
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def main():
    d = pd.read_csv(os.path.join(DATA, "combined_fake_reviews.csv"))
    p = d[(d.split == "test") & (d.domain == "products")].copy()
    p["text"] = p["text"].astype(str)
    p = p[p.text.str.split().str.len() >= 5]                 # drop trivially short
    # shuffle so fake/genuine are interleaved (the browse loads sequential slices)
    p = p.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out = pd.DataFrame({
        "review_id":          [f"amz_{i}" for i in range(len(p))],
        "review_text":        p["text"].values,
        "ground_truth_label": p["label"].astype(int).values,   # 1 = fake (AI), 0 = genuine (real)
        "source":             p["source"].values,              # generator name or 'amazon_real'
    })
    out.to_csv(os.path.join(DATA, "amazon_fake_labeled.csv"), index=False)
    print(f"wrote data/amazon_fake_labeled.csv: {len(out)} reviews "
          f"({int((out.ground_truth_label==1).sum())} fake / {int((out.ground_truth_label==0).sum())} genuine)")


if __name__ == "__main__":
    main()

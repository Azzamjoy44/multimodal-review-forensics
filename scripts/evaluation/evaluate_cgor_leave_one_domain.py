# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
evaluate_cgor_leave_one_domain.py
---------------------------------
Leave-one-DOMAIN-out evaluation for the broadened, multi-domain CG/OR detector
("Model A"). Companion to evaluate_cgor_generalization.py (which holds out
GENERATORS); this holds out whole DOMAINS.

The question: if the detector has never seen reviews from a domain (e.g. it was
trained on products/movies/restaurants/apps/games but NOT books), does it still
catch AI-generated fakes in that unseen domain? High leave-one-domain recall =
the model learned a domain-transferable "machine-written" signal, not a per-domain
trick. This is the core "works on all kinds of reviews users upload" metric.

For each domain D:
    * in-dist recall    = model trained on ALL data, evaluated on D's test CG
    * LOD recall        = model trained with domain D fully EXCLUDED, on D's test CG
    * LOD specificity   = that same model's accuracy on D's held-out human (OR) reviews
Uses Logistic Regression (fast, representative) as the probe, matching the
generator-LOGO script.

    python evaluate_cgor_leave_one_domain.py
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA = "data/fake_reviews_broadened.csv"


def fit(texts, labels):
    vec = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2, stop_words="english")
    X = vec.fit_transform(texts)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=42).fit(X, labels)
    return vec, lr


def main():
    df = pd.read_csv(DATA)
    df["text"] = df["text"].astype(str)
    if "domain" not in df.columns:
        raise SystemExit("fake_reviews_broadened.csv has no 'domain' column — rebuild with the multi-domain builder.")

    trainpool = df[df.split.isin(["train", "val"])]
    test      = df[df.split == "test"]
    domains   = sorted(df.domain.unique())

    vec0, lr0 = fit(trainpool.text, trainpool.label)
    overall = (lr0.predict(vec0.transform(test.text)) == test.label).mean()
    print(f"baseline (train on ALL): overall test acc={overall:.3f}\n")

    print(f"{'held-out domain':16}{'n_test_CG':>11}{'in-dist recall':>16}"
          f"{'LOD recall':>13}{'LOD specificity':>17}")
    print("-" * 73)
    lod_recalls = []
    for D in domains:
        dtest_cg = test[(test.label == 1) & (test.domain == D)]
        dtest_or = test[(test.label == 0) & (test.domain == D)]
        if len(dtest_cg) == 0:
            continue
        # in-distribution: model that HAS seen domain D
        indist = (lr0.predict(vec0.transform(dtest_cg.text)) == 1).mean()
        # leave-one-domain-out: model that has NOT seen domain D
        pool = trainpool[trainpool.domain != D]
        vecD, lrD = fit(pool.text, pool.label)
        lod_rec = (lrD.predict(vecD.transform(dtest_cg.text)) == 1).mean()
        lod_spec = (lrD.predict(vecD.transform(dtest_or.text)) == 0).mean() if len(dtest_or) else float("nan")
        lod_recalls.append(lod_rec)
        print(f"{D:16}{len(dtest_cg):>11}{indist:>15.3f}{lod_rec:>13.3f}{lod_spec:>17.3f}")
    print("-" * 73)
    print(f"mean leave-one-domain recall (unseen-domain catch rate): "
          f"{sum(lod_recalls)/len(lod_recalls):.3f}")
    print("\nLOD recall = fraction of an UNSEEN domain's AI reviews still caught as machine.")
    print("Big in-dist -> LOD drop = the model leaned on domain-specific cues, not general machine-ness.")


if __name__ == "__main__":
    main()

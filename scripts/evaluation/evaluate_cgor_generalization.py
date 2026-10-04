# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
evaluate_cgor_generalization.py
-------------------------------
Leave-one-GENERATOR-out (LOGO) evaluation for the broadened CG/OR detector.

The in-distribution F1 is misleading (it rewards memorising each generator's
fingerprint). The real "how close to catch-all" metric is: train on all generators
EXCEPT one, then test whether that *unseen* generator's reviews are still caught as
machine-generated.

For each machine-generator (CG source) G:
    * in-dist recall = a model trained on ALL data, evaluated on G's test reviews
    * LOGO recall    = a model trained with G fully EXCLUDED, evaluated on G's reviews
    * LOGO specificity = that same model's accuracy on held-out human (OR) reviews
A big in-dist -> LOGO drop means the model relies on G's fingerprint, not general
machine-ness. Uses Logistic Regression (fast, representative) as the probe.

Re-run this after adding generators (Route B) to see generalization improve.

    python evaluate_cgor_generalization.py
"""

import numpy as np
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
    trainpool = df[df.split.isin(["train", "val"])]
    test      = df[df.split == "test"]
    or_test   = test[test.label == 0]
    cg_sources = sorted(df[df.label == 1].source.unique())

    vec0, lr0 = fit(trainpool.text, trainpool.label)
    overall = (lr0.predict(vec0.transform(test.text)) == test.label).mean()
    spec0   = (lr0.predict(vec0.transform(or_test.text)) == 0).mean()
    print(f"baseline (train on ALL):  overall test acc={overall:.3f}  "
          f"human specificity={spec0:.3f}\n")

    print(f"{'generator':16}{'n_test':>8}{'in-dist recall':>16}"
          f"{'LOGO recall':>14}{'LOGO specificity':>18}")
    print("-" * 72)
    logo_recalls = []
    for G in cg_sources:
        gtest = test[(test.label == 1) & (test.source == G)]
        if len(gtest) == 0:
            continue
        ind = (lr0.predict(vec0.transform(gtest.text)) == 1).mean()
        logo_train = trainpool[~((trainpool.label == 1) & (trainpool.source == G))]
        vec, lr = fit(logo_train.text, logo_train.label)
        logo_rec  = (lr.predict(vec.transform(gtest.text)) == 1).mean()
        logo_spec = (lr.predict(vec.transform(or_test.text)) == 0).mean()
        logo_recalls.append(logo_rec)
        print(f"{G:16}{len(gtest):>8}{ind:>16.3f}{logo_rec:>14.3f}{logo_spec:>18.3f}")

    print("-" * 72)
    if logo_recalls:
        print(f"mean LOGO recall (unseen-generator catch rate): {np.mean(logo_recalls):.3f}")
    print("\nLOGO recall = fraction of an UNSEEN generator's reviews caught as machine "
          "(higher = better generalization; in-dist - LOGO gap = fingerprint reliance).")


if __name__ == "__main__":
    main()

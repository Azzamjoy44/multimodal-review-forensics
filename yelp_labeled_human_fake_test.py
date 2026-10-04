"""
yelp_labeled_human_fake_test.py — OUT-OF-DISTRIBUTION labeled test of the fake detectors on
real-world human-made fake reviews they NEVER trained on (Yelp).

The human-deception (Ott+Li) and CG (Model A) detectors never saw Yelp; this measures how
their TEXT-based deception signal transfers to Yelp's labeled human spam. Reports accuracy
metrics (macro-F1, fake recall, genuine specificity) against the true Yelp labels — unlike
the Amazon test, here we HAVE ground truth.

Honest expectation: Yelp's fakes are BEHAVIOURALLY labelled (its spam filter) and are
textually near-indistinguishable (~69% ceiling even for Yelp-TRAINED text models), and are a
*different kind* of human fake than Ott/Li's elicited deception — so limited transfer is
expected and itself informative ("human fake" is not one thing).

Detectors: CG/Model A · human-deception (Ott+Li) · UNIFIED (CG+human). sklearn-majority + DistilBERT.
Caveat: the UNIFIED model's CG/OR training pool contained some genuine Yelp reviews ("rd_yelp")
as OR examples, so its Yelp number may be mildly optimistic; the human-only model is clean OOD.

RUN:  python yelp_labeled_human_fake_test.py [sample]   (default 2500)
"""

import os
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_TORCH", "1")
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, recall_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
YELP = os.path.join(HERE, "data", "yelp_split.csv")
SK_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500


def sklearn_majority(model_dir, texts):
    thr = json.load(open(os.path.join(model_dir, "thresholds.json")))
    cols = []
    for nme in SK_NAMES:
        p = os.path.join(model_dir, f"{nme}.joblib")
        if os.path.exists(p):
            proba = joblib.load(p).predict_proba(texts)[:, 1]
            cols.append((proba >= float(thr.get(nme, 0.5))).astype(int))
    V = np.vstack(cols)
    return (V.sum(0) > V.shape[0] / 2).astype(int)


def hf_distilbert(model_dir, texts, batch=32):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir); model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s+batch], truncation=True, padding=True, max_length=256, return_tensors="pt")
            enc.pop("token_type_ids", None)          # DistilBERT has no segment embeddings
            out += model(**enc).logits.argmax(1).tolist()
    return np.array(out)


def report(name, y, pred):
    return (name, f1_score(y, pred, average="macro", zero_division=0),
            recall_score(y, pred, pos_label=1, zero_division=0),
            recall_score(y, pred, pos_label=0, zero_division=0))


def main():
    df = pd.read_csv(YELP); df["text"] = df["text"].astype(str)
    te = df[df.split == "test"] if "split" in df.columns else df
    te = te.sample(min(N, len(te)), random_state=42)
    texts, y = te.text.tolist(), te.label.values.astype(int)
    print(f"Yelp labeled test sample: {len(texts):,} ({int((y==1).sum())} fake / {int((y==0).sum())} genuine)\n", flush=True)

    rows = []
    for label, sk_dir in [
        ("CG / Model A",       os.path.join(HERE, "data", "fake_broadened_sklearn_models")),
        ("human-deception",    os.path.join(HERE, "data", "human_fake", "human_fake_sklearn_models")),
        ("UNIFIED (CG+human)", os.path.join(HERE, "data", "combined_fake_models")),
    ]:
        rows.append(report(label + " [sklearn]", y, sklearn_majority(sk_dir, texts)))
        print(f"  {label} sklearn done", flush=True)

    print("  loading DistilBERTs ...", flush=True)
    try:
        import model_a
        cg = np.array([1 if (d or {}).get("label") == 1 else 0 for d in model_a.score_fake_distilbert(texts)])
        rows.append(report("CG / Model A [DistilBERT]", y, cg))
    except Exception as e:
        print(f"  (CG DistilBERT skipped: {e})")
    for label, ddir in [("human-deception [DistilBERT]", os.path.join(HERE, "data", "human_fake", "human_fake_distilbert")),
                        ("UNIFIED (CG+human) [DistilBERT]", os.path.join(HERE, "data", "combined_fake_distilbert"))]:
        try:
            rows.append(report(label, y, hf_distilbert(ddir, texts)))
        except Exception as e:
            print(f"  ({label} skipped: {e})")

    print("\n" + "=" * 70)
    print(f"OOD LABELED TEST — Yelp human fakes (n={len(y):,}); ground-truth metrics")
    print("=" * 70)
    print(f"{'detector':<34}{'macroF1':>9}{'fakeR':>8}{'genuineSpec':>13}")
    for name, mf1, fr, ge in rows:
        print(f"{name:<34}{mf1:>8.1%}{fr:>8.1%}{ge:>12.1%}")
    print("\nReference (in-distribution, Ott+Li test): human LR 88.2% / DistilBERT 87.6%.")
    print("Yelp = behaviourally-labelled, textually ~indistinguishable human spam (~69% ceiling")
    print("even for Yelp-trained text models) -> limited transfer expected, and itself informative.")


if __name__ == "__main__":
    main()

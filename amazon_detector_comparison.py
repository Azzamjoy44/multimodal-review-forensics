"""
amazon_detector_comparison.py — apply the three fake-detector families to the same sample of
real Amazon Fine Food reviews and compare their FLAG RATE (% fired).

  * CG / Model A  (data/fake_broadened_sklearn_models + Model A DistilBERT) — AI/computer-generated detector
  * human         (data/human_fake/human_fake_sklearn_models + .../human_fake_distilbert) — human-deception
  * unified       (data/combined_fake_models + data/combined_fake_distilbert) — CG OR human

IMPORTANT: Amazon Fine Foods has NO fake labels, and these ~2007-2012 reviews are
overwhelmingly genuine human writing (pre-LLM). So this is a FLAG-RATE comparison, not an
accuracy one: a LOW rate = the detector is appropriately quiet (correct for CG, since there's
no AI text); a HIGH rate from the human/unified detector = mostly FALSE POSITIVES / weak
cross-domain transfer (these detectors trained on hotel/restaurant/doctor MTurk deception,
applied here to food reviews). Reports sklearn-majority + DistilBERT flag rate per family.

RUN:  python amazon_detector_comparison.py [sample]   (default 2000)
"""

import os
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_TORCH", "1")  # HF DistilBERT: torch only
import sys
import csv
import json
import random
import warnings

import numpy as np
import joblib

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)
REVIEWS = os.path.join(HERE, "data", "Reviews.csv")
SK_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000


def load_sample(n):
    rows = []
    with open(REVIEWS, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            t = (r.get("Text") or "").strip()
            if t:
                rows.append(t)
    random.seed(42)
    return random.sample(rows, min(n, len(rows)))


def sklearn_majority_flag(model_dir, texts):
    thr = json.load(open(os.path.join(model_dir, "thresholds.json")))
    votes = np.zeros((len(texts), 0), dtype=int)
    cols = []
    for nme in SK_NAMES:
        p = os.path.join(model_dir, f"{nme}.joblib")
        if not os.path.exists(p):
            continue
        proba = joblib.load(p).predict_proba(texts)[:, 1]
        cols.append((proba >= float(thr.get(nme, 0.5))).astype(int))
    V = np.vstack(cols)                         # [n_models, n]
    maj = (V.sum(0) > V.shape[0] / 2).astype(int)
    return maj.mean(), V.mean(1)                # majority flag rate, per-model flag rates


def hf_distilbert_flag(model_dir, texts, batch=32):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir); model.eval()
    flags = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s+batch], truncation=True, padding=True, max_length=256, return_tensors="pt")
            enc.pop("token_type_ids", None)          # DistilBERT has no segment embeddings
            flags += model(**enc).logits.argmax(1).tolist()
    return float(np.mean(flags))


def main():
    texts = load_sample(N)
    print(f"Amazon Fine Foods sample: {len(texts):,} reviews (seed 42)\n")

    results = []
    for label, sk_dir in [
        ("CG / Model A", os.path.join(HERE, "data", "fake_broadened_sklearn_models")),
        ("human-deception", os.path.join(HERE, "data", "human_fake", "human_fake_sklearn_models")),
        ("UNIFIED (CG+human)", os.path.join(HERE, "data", "combined_fake_models")),
    ]:
        maj, _ = sklearn_majority_flag(sk_dir, texts)
        results.append([label, maj, None])
        print(f"  {label:<20} sklearn-majority flagged fake: {maj:.1%}", flush=True)

    # DistilBERTs
    print("\n  loading DistilBERTs ...", flush=True)
    db = {}
    try:
        import model_a
        db["CG / Model A"] = float(np.mean([1 if (d or {}).get("label") == 1 else 0
                                            for d in model_a.score_fake_distilbert(texts)]))
    except Exception as e:
        print(f"  (Model A DistilBERT skipped: {e})")
    for label, ddir in [("human-deception", os.path.join(HERE, "data", "human_fake", "human_fake_distilbert")),
                        ("UNIFIED (CG+human)", os.path.join(HERE, "data", "combined_fake_distilbert"))]:
        try:
            db[label] = hf_distilbert_flag(ddir, texts)
        except Exception as e:
            print(f"  ({label} DistilBERT skipped: {e})")
    for r in results:
        r[2] = db.get(r[0])

    print("\n" + "=" * 62)
    print(f"FLAG RATE on {len(texts):,} real (unlabeled, ~genuine) Amazon reviews")
    print("=" * 62)
    print(f"{'detector':<22}{'sklearn-majority':>18}{'DistilBERT':>14}")
    for label, maj, dbf in results:
        print(f"{label:<22}{maj:>17.1%}{('  n/a' if dbf is None else f'{dbf:.1%}'):>14}")
    print("\nLow = appropriately quiet; high (human/unified) = likely false positives /")
    print("weak cross-domain transfer (no AI text + reviews are mostly genuine).")


if __name__ == "__main__":
    main()

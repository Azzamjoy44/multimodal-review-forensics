"""
evaluate_yelp_focal.py
-----------------------
Compares focal loss vs baseline for LSTM and DistilBERT on the Yelp test split.

Usage:
    python evaluate_yelp_focal.py

Requires the four model directories to be present under data/:
    data/yelp_fake_lstm_onnx/          (baseline LSTM)
    data/yelp_fake_lstm_focal_onnx/    (focal LSTM)
    data/yelp_fake_distilbert_onnx/    (baseline DistilBERT)
    data/yelp_fake_distilbert_focal_onnx/  (focal DistilBERT)
"""

import os, re, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

DATA_DIR            = os.path.join(os.path.dirname(__file__), "data")
LSTM_BASE_DIR       = os.path.join(DATA_DIR, "yelp_fake_lstm_onnx")
LSTM_FOCAL_DIR      = os.path.join(DATA_DIR, "yelp_fake_lstm_focal_onnx")
BERT_BASE_DIR       = os.path.join(DATA_DIR, "yelp_fake_distilbert_onnx")
BERT_FOCAL_DIR      = os.path.join(DATA_DIR, "yelp_fake_distilbert_focal_onnx")
SPLIT_CSV           = os.path.join(DATA_DIR, "yelp_split.csv")
IMBALANCED_CSV      = os.path.join(DATA_DIR, "yelp_split_imbalanced.csv")
LSTM_MAX_LEN        = 250   # must match training (MAX_LEN=250 in the LSTM notebooks)
BATCH_SIZE          = 64


def _norm(text):
    """Lowercase + collapse whitespace — same normalization used during split creation."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Load test split
# ---------------------------------------------------------------------------
print("Loading test split...")
df = pd.read_csv(SPLIT_CSV)
test_df = df[df["split"] == "test"].reset_index(drop=True)
X_test  = test_df["text"].tolist()
y_test  = test_df["label"].tolist()
print(f"  Test rows : {len(X_test):,}  (genuine={sum(1 for y in y_test if y==0):,}  fake={sum(1 for y in y_test if y==1):,})\n")


# ---------------------------------------------------------------------------
# Leakage check — focal training data must not contain any test-split reviews
# ---------------------------------------------------------------------------
print("=" * 60)
print("Leakage check")
print("=" * 60)

if not os.path.exists(IMBALANCED_CSV):
    print(f"  WARNING: {IMBALANCED_CSV} not found — skipping leakage check.")
    print("  Generate it by running prepare_yelp_split_imbalanced.ipynb on Colab.")
    print()
else:
    print(f"  Normalizing {len(X_test):,} test texts from yelp_split.csv ...")
    test_norms = set(_norm(t) for t in X_test)

    print(f"  Loading training texts from yelp_split_imbalanced.csv ...")
    imb_df      = pd.read_csv(IMBALANCED_CSV)
    train_texts = imb_df[imb_df["split"] == "train"]["text"].tolist()
    print(f"  Normalizing {len(train_texts):,} focal training texts ...")
    train_norms = set(_norm(t) for t in train_texts)

    overlap     = test_norms & train_norms
    n_overlap   = len(overlap)
    pct_overlap = n_overlap / len(test_norms) * 100

    print(f"\n  Balanced test texts  : {len(test_norms):,}")
    print(f"  Focal training texts : {len(train_norms):,}")
    print(f"  Overlapping texts    : {n_overlap:,}  ({pct_overlap:.3f}% of test set)")

    if n_overlap == 0:
        print("\n  PASS — No overlap. Focal training data is clean w.r.t. the test set.")
        print("  Comparison on yelp_split.csv test set is valid.\n")
    else:
        print(f"\n  FAIL — {n_overlap:,} test reviews ({pct_overlap:.2f}%) appear in the focal")
        print("  training data. The comparison on yelp_split.csv test set is INVALID.")
        print("  Regenerate yelp_split_imbalanced.csv using prepare_yelp_split_imbalanced.ipynb.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def row(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    fp   = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    fr   = recall_score   (y_true, y_pred, pos_label=1, zero_division=0)
    ff1  = f1_score       (y_true, y_pred, pos_label=1, zero_division=0)
    gp   = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    gr   = recall_score   (y_true, y_pred, pos_label=0, zero_division=0)
    gf1  = f1_score       (y_true, y_pred, pos_label=0, zero_division=0)
    return {
        "Model":       name,
        "Accuracy":    f"{acc:.1%}",
        "Macro F1":    f"{mf1:.1%}",
        "Fake P":      f"{fp:.1%}",
        "Fake R":      f"{fr:.1%}",
        "Fake F1":     f"{ff1:.1%}",
        "Genuine P":   f"{gp:.1%}",
        "Genuine R":   f"{gr:.1%}",
        "Genuine F1":  f"{gf1:.1%}",
        "_macro_f1":   mf1,
    }


def encode_lstm(texts, tok):
    seqs = tok.texts_to_sequences(texts)
    padded = []
    for s in seqs:
        s = s[:LSTM_MAX_LEN]
        s += [0] * (LSTM_MAX_LEN - len(s))
        padded.append(s)
    return np.array(padded, dtype=np.float32)


def run_lstm(model_dir, label):
    onnx_path  = os.path.join(model_dir, os.path.basename(model_dir).replace("_onnx", "") + ".onnx")
    # also try just 'model.onnx' as fallback
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(model_dir, "model.onnx")
    tok_path   = os.path.join(model_dir, os.path.basename(model_dir).replace("_onnx", "") + "_tokenizer.json")
    if not os.path.exists(tok_path):
        # look for any *tokenizer.json file
        for fn in os.listdir(model_dir):
            if fn.endswith("_tokenizer.json"):
                tok_path = os.path.join(model_dir, fn)
                break
    thresh_path = os.path.join(model_dir, os.path.basename(model_dir).replace("_onnx", "") + "_threshold.json")
    if not os.path.exists(thresh_path):
        thresh_path = os.path.join(model_dir, "threshold.json")

    if not os.path.exists(onnx_path):
        print(f"  [{label}] MISSING ({onnx_path}) — skipping")
        return None

    import onnxruntime as ort
    from score_reviews import _KerasTokenizer

    lstm_tok   = _KerasTokenizer.from_json(tok_path)
    threshold  = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            threshold = json.load(f).get("threshold", 0.5)

    sess        = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    all_probs   = []
    for i in range(0, len(X_test), BATCH_SIZE):
        batch = encode_lstm(X_test[i:i+BATCH_SIZE], lstm_tok)
        out   = sess.run(None, {input_name: batch})[0].flatten()
        all_probs.extend(out.tolist())

    y_pred = (np.array(all_probs) >= threshold).astype(int)
    r = row(label, y_test, y_pred)
    print(f"  [{label}] Macro F1={r['Macro F1']}  threshold={threshold:.2f}")
    return r


def run_distilbert(model_dir, label):
    if not os.path.isdir(model_dir):
        print(f"  [{label}] MISSING ({model_dir}) — skipping")
        return None

    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import pipeline as hf_pipeline

    thresh_path   = os.path.join(model_dir, "threshold.json")
    bert_threshold = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            bert_threshold = json.load(f).get("threshold", 0.5)

    tokenizer  = AutoTokenizer.from_pretrained(model_dir)
    ort_model  = ORTModelForSequenceClassification.from_pretrained(
        model_dir, provider="CPUExecutionProvider"
    )
    bert_pipe  = hf_pipeline(
        "text-classification", model=ort_model, tokenizer=tokenizer,
        device=-1, batch_size=8, truncation=True, max_length=512,
    )

    all_probs = []
    for i in range(0, len(X_test), BATCH_SIZE):
        batch   = X_test[i:i+BATCH_SIZE]
        results = bert_pipe(batch)
        for res in results:
            score     = res["score"]
            prob_fake = score if res["label"] in ("LABEL_1", "fake", "1") else 1 - score
            all_probs.append(prob_fake)

    y_pred = (np.array(all_probs) >= bert_threshold).astype(int)
    r = row(label, y_test, y_pred)
    print(f"  [{label}] Macro F1={r['Macro F1']}  threshold={bert_threshold:.2f}")
    return r


# ---------------------------------------------------------------------------
# Run all four models
# ---------------------------------------------------------------------------
rows = []

print("=" * 60)
print("Evaluating LSTM models...")
print("=" * 60)
for model_dir, label in [
    (LSTM_BASE_DIR,  "LSTM (baseline)"),
    (LSTM_FOCAL_DIR, "LSTM (focal γ=2)"),
]:
    r = run_lstm(model_dir, label)
    if r:
        rows.append(r)

print()
print("=" * 60)
print("Evaluating DistilBERT models...")
print("=" * 60)
for model_dir, label in [
    (BERT_BASE_DIR,  "DistilBERT (baseline)"),
    (BERT_FOCAL_DIR, "DistilBERT (focal γ=2)"),
]:
    r = run_distilbert(model_dir, label)
    if r:
        rows.append(r)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 100)
print("FOCAL LOSS ABLATION — Yelp Test Set  (n={:,}  genuine={:,}  fake={:,})".format(
    len(y_test),
    sum(1 for y in y_test if y == 0),
    sum(1 for y in y_test if y == 1),
))
print("=" * 100)

if not rows:
    print("  No models evaluated.")
else:
    results_df  = pd.DataFrame(rows)
    display_df  = results_df.drop(columns=["_macro_f1"]).set_index("Model")
    col_w       = 12
    headers     = ["Model"] + list(display_df.columns)
    widths      = [24] + [col_w] * len(display_df.columns)

    print("".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for _, r in results_df.sort_values("_macro_f1", ascending=False).iterrows():
        vals = [r["Model"]] + [r[c] for c in display_df.columns]
        print("".join(v.rjust(w) for v, w in zip(vals, widths)))

    print("=" * 100)
    print("Columns: Accuracy | Macro F1 | Fake P | Fake R | Fake F1 | Genuine P | Genuine R | Genuine F1")

    # Delta summary
    print()
    baseline_rows = {r["Model"]: r for r in rows}
    pairs = [
        ("LSTM (baseline)",       "LSTM (focal γ=2)"),
        ("DistilBERT (baseline)", "DistilBERT (focal γ=2)"),
    ]
    print("Delta summary (focal − baseline):")
    for base_name, focal_name in pairs:
        if base_name in baseline_rows and focal_name in baseline_rows:
            b = baseline_rows[base_name]
            f = baseline_rows[focal_name]
            delta_f1  = f["_macro_f1"] - b["_macro_f1"]
            sign      = "+" if delta_f1 >= 0 else ""
            print(f"  {focal_name:<30}  ΔMacro F1 = {sign}{delta_f1:.1%}")

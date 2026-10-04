"""
evaluate_yelp_models.py
-----------------------
Evaluates all 10 Yelp fake detection models on the held-out test split.

Usage:
    python evaluate_yelp_models.py

Output: per-model table (accuracy, macro F1, fake precision/recall/F1,
        genuine precision/recall/F1) + a summary ranking by macro F1.
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

DATA_DIR         = os.path.join(os.path.dirname(__file__), "data")
SKLEARN_DIR      = os.path.join(DATA_DIR, "yelp_fake_sklearn_models")
LSTM_DIR         = os.path.join(DATA_DIR, "yelp_fake_lstm_onnx")
DISTILBERT_DIR   = os.path.join(DATA_DIR, "yelp_fake_distilbert_onnx")
SPLIT_CSV        = os.path.join(DATA_DIR, "yelp_split.csv")
SKLEARN_NAMES    = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
LSTM_MAX_LEN     = 250   # must match training (MAX_LEN=250 in the LSTM notebooks)
BATCH_SIZE       = 64


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
# Helpers
# ---------------------------------------------------------------------------
def row(name, y_true, y_pred):
    acc    = accuracy_score(y_true, y_pred)
    mf1    = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    fp     = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    fr     = recall_score   (y_true, y_pred, pos_label=1, zero_division=0)
    ff1    = f1_score       (y_true, y_pred, pos_label=1, zero_division=0)
    gp     = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    gr     = recall_score   (y_true, y_pred, pos_label=0, zero_division=0)
    gf1    = f1_score       (y_true, y_pred, pos_label=0, zero_division=0)
    return {
        "Model":        name,
        "Accuracy":     f"{acc:.1%}",
        "Macro F1":     f"{mf1:.1%}",
        "Fake P":       f"{fp:.1%}",
        "Fake R":       f"{fr:.1%}",
        "Fake F1":      f"{ff1:.1%}",
        "Genuine P":    f"{gp:.1%}",
        "Genuine R":    f"{gr:.1%}",
        "Genuine F1":   f"{gf1:.1%}",
        "_macro_f1":    mf1,
    }


# ---------------------------------------------------------------------------
# sklearn models
# ---------------------------------------------------------------------------
import joblib

thresh_path = os.path.join(SKLEARN_DIR, "yelp_fake_thresholds.json")
with open(thresh_path) as f:
    thresholds = json.load(f)

rows = []
print("=" * 56)
print("Evaluating sklearn models...")
print("=" * 56)

for name in SKLEARN_NAMES:
    path = os.path.join(SKLEARN_DIR, f"yelp_fake_{name}.joblib")
    if not os.path.exists(path):
        print(f"  [{name.upper()}] MISSING — skipping")
        continue
    print(f"  [{name.upper()}]", end=" ", flush=True)
    try:
        pipeline = joblib.load(path)
    except Exception as e:
        print(f"LOAD ERROR — {e}")
        continue
    threshold = thresholds.get(name, 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "knn":
            # KNN OOMs on full 25k test set — run in chunks
            chunk, all_probs = 500, []
            for i in range(0, len(X_test), chunk):
                all_probs.extend(pipeline.predict_proba(X_test[i:i+chunk])[:, 1].tolist())
            probs = np.array(all_probs)
        else:
            probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)
    r = row(name.upper(), y_test, y_pred)
    print(f"Macro F1={r['Macro F1']}  threshold={threshold:.2f}")
    rows.append(r)


# ---------------------------------------------------------------------------
# LSTM ONNX
# ---------------------------------------------------------------------------
print("\nEvaluating LSTM...")
onnx_path   = os.path.join(LSTM_DIR, "yelp_fake_lstm.onnx")
tok_path    = os.path.join(LSTM_DIR, "yelp_fake_lstm_tokenizer.json")
thresh_path = os.path.join(LSTM_DIR, "yelp_fake_lstm_threshold.json")

if not os.path.exists(onnx_path):
    print("  LSTM MISSING — skipping")
else:
    import onnxruntime as ort
    from score_reviews import _KerasTokenizer

    lstm_tok = _KerasTokenizer.from_json(tok_path)

    def encode_lstm(texts):
        seqs = lstm_tok.texts_to_sequences(texts)
        padded = []
        for s in seqs:
            s = s[:LSTM_MAX_LEN]
            s += [0] * (LSTM_MAX_LEN - len(s))
            padded.append(s)
        return np.array(padded, dtype=np.float32)

    lstm_threshold = 0.57
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            lstm_threshold = json.load(f).get("threshold", 0.57)

    sess        = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name  = sess.get_inputs()[0].name
    all_probs   = []
    for i in range(0, len(X_test), BATCH_SIZE):
        batch = encode_lstm(X_test[i:i+BATCH_SIZE])
        out   = sess.run(None, {input_name: batch})[0].flatten()
        all_probs.extend(out.tolist())

    y_pred = (np.array(all_probs) >= lstm_threshold).astype(int)
    r = row("LSTM", y_test, y_pred)
    print(f"  Macro F1={r['Macro F1']}  threshold={lstm_threshold:.2f}")
    rows.append(r)


# ---------------------------------------------------------------------------
# DistilBERT ONNX
# ---------------------------------------------------------------------------
print("\nEvaluating DistilBERT...")
if not os.path.isdir(DISTILBERT_DIR):
    print("  DistilBERT MISSING — skipping")
else:
    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import pipeline as hf_pipeline

    thresh_path = os.path.join(DISTILBERT_DIR, "threshold.json")
    bert_threshold = 0.51
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            bert_threshold = json.load(f).get("threshold", 0.51)

    tokenizer    = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    ort_model    = ORTModelForSequenceClassification.from_pretrained(
        DISTILBERT_DIR, provider="CPUExecutionProvider"
    )
    bert_pipe    = hf_pipeline(
        "text-classification", model=ort_model, tokenizer=tokenizer,
        device=-1, batch_size=8, truncation=True, max_length=512,
    )

    all_probs = []
    for i in range(0, len(X_test), BATCH_SIZE):
        batch   = X_test[i:i+BATCH_SIZE]
        results = bert_pipe(batch)
        for res in results:
            label = res["label"]   # e.g. "LABEL_1" or "LABEL_0"
            score = res["score"]
            prob_fake = score if label in ("LABEL_1", "fake", "1") else 1 - score
            all_probs.append(prob_fake)

    y_pred = (np.array(all_probs) >= bert_threshold).astype(int)
    r = row("DistilBERT", y_test, y_pred)
    print(f"  Macro F1={r['Macro F1']}  threshold={bert_threshold:.2f}")
    rows.append(r)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 90)
print("YELP FAKE DETECTION — Test Set Results  (n={:,}  genuine={:,}  fake={:,})".format(
    len(y_test),
    sum(1 for y in y_test if y == 0),
    sum(1 for y in y_test if y == 1),
))
print("=" * 90)

results_df = pd.DataFrame(rows)
display_df = results_df.drop(columns=["_macro_f1"]).set_index("Model")

col_w   = 11
headers = ["Model"] + list(display_df.columns)
widths  = [12] + [col_w] * len(display_df.columns)

# Header
header_line = "".join(h.rjust(w) for h, w in zip(headers, widths))
print(header_line)
print("-" * sum(widths))

# Rows sorted by macro F1 descending
for _, r in results_df.sort_values("_macro_f1", ascending=False).iterrows():
    vals = [r["Model"]] + [r[c] for c in display_df.columns]
    print("".join(v.rjust(w) for v, w in zip(vals, widths)))

print("=" * 90)
print("Columns: Accuracy | Macro F1 | Fake Precision | Fake Recall | Fake F1 |"
      " Genuine Precision | Genuine Recall | Genuine F1")
print()

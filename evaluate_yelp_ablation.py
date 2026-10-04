"""
evaluate_yelp_ablation.py
--------------------------
Compares baseline vs focal vs contrastive for LSTM and DistilBERT
on the Yelp balanced test split.

Fair-comparison design:
  - Each model's decision threshold is RE-TUNED on the balanced yelp_split.csv
    VAL split (not the threshold it shipped with, which was tuned on a different
    class balance). This removes the threshold-regime confound: focal/contrastive
    ship thresholds tuned on ~21%-fake val, the baseline on 50/50 val — applying
    those as-is to the balanced test would unfairly understate focal/contrastive.
  - PR-AUC and ROC-AUC are also reported. These are THRESHOLD-FREE: they measure
    how well a model ranks fakes above genuine across all thresholds, so they are
    the cleanest "is it a good detector" numbers, immune to threshold choice.

No leakage: focal/contrastive exclude yelp_split.csv val/test from training
(pinned), and the baseline holds its val out of training — so tuning on the
balanced val is valid for every model.

Usage:
    python evaluate_yelp_ablation.py

Requires model directories under data/:
    data/yelp_fake_lstm_onnx/                    (baseline LSTM)
    data/yelp_fake_lstm_focal_onnx/              (focal LSTM)
    data/yelp_fake_lstm_contrastive_onnx/        (contrastive LSTM)
    data/yelp_fake_distilbert_onnx/              (baseline DistilBERT)
    data/yelp_fake_distilbert_focal_onnx/        (focal DistilBERT)
    data/yelp_fake_distilbert_contrastive_onnx/  (contrastive DistilBERT)
"""

import os, re, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score, roc_auc_score,
)

# Real-world fake rate on Yelp (~13% after dedup). The test set is balanced 50/50,
# but recall and false-positive rate are prior-independent, so we can project the
# precision a model would actually achieve in deployment at this prior. This is
# what matters for "is it a good fake detector", not balanced-test macro F1.
REAL_WORLD_FAKE_RATE = 0.13

DATA_DIR                = os.path.join(os.path.dirname(__file__), "data")
LSTM_BASE_DIR           = os.path.join(DATA_DIR, "yelp_fake_lstm_onnx")
LSTM_FOCAL_DIR          = os.path.join(DATA_DIR, "yelp_fake_lstm_focal_onnx")
LSTM_CONTRASTIVE_DIR    = os.path.join(DATA_DIR, "yelp_fake_lstm_contrastive_onnx")
BERT_BASE_DIR           = os.path.join(DATA_DIR, "yelp_fake_distilbert_onnx")
BERT_FOCAL_DIR          = os.path.join(DATA_DIR, "yelp_fake_distilbert_focal_onnx")
BERT_CONTRASTIVE_DIR    = os.path.join(DATA_DIR, "yelp_fake_distilbert_contrastive_onnx")
SPLIT_CSV               = os.path.join(DATA_DIR, "yelp_split.csv")
IMBALANCED_CSV          = os.path.join(DATA_DIR, "yelp_split_imbalanced.csv")
LSTM_MAX_LEN            = 250   # must match training (MAX_LEN=250 in the LSTM notebooks)
BERT_MAX_LEN            = 256   # must match training (MAX_LEN=256 in the DistilBERT notebooks)
BATCH_SIZE              = 64


# ---------------------------------------------------------------------------
# Preflight — fail fast if any required input is missing (report ALL at once)
# ---------------------------------------------------------------------------
_required_files = [SPLIT_CSV, IMBALANCED_CSV]
_required_dirs  = [
    LSTM_BASE_DIR, LSTM_FOCAL_DIR, LSTM_CONTRASTIVE_DIR,
    BERT_BASE_DIR, BERT_FOCAL_DIR, BERT_CONTRASTIVE_DIR,
]

_missing = []
for _f in _required_files:
    if not os.path.isfile(_f):
        _missing.append(f"  [missing file]    {_f}")
for _d in _required_dirs:
    if not os.path.isdir(_d):
        _missing.append(f"  [missing folder]  {_d}")
    elif not any(fn.endswith(".onnx") for fn in os.listdir(_d)):
        _missing.append(f"  [no .onnx inside] {_d}")

if _missing:
    print("=" * 60)
    print("ERROR — required inputs missing; nothing was evaluated.")
    print("=" * 60)
    print("\n".join(_missing))
    print("\nPlace the deduped split CSVs and all six model folders under data/, then re-run.")
    print("Expected:")
    print(f"  {SPLIT_CSV}")
    print(f"  {IMBALANCED_CSV}")
    for _d in _required_dirs:
        print(f"  {_d}{os.sep}")
    sys.exit(1)

print("Preflight OK — all required inputs present.\n")


def _norm(text):
    """Lowercase + collapse whitespace — same normalisation used during split creation."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Load balanced val + test splits
# ---------------------------------------------------------------------------
print("Loading balanced val + test splits...")
df      = pd.read_csv(SPLIT_CSV)
val_df  = df[df["split"] == "val" ].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)
X_val,  y_val  = val_df["text"].tolist(),  val_df["label"].tolist()
X_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()
print(f"  Val  rows : {len(X_val):,}  "
      f"(genuine={sum(1 for y in y_val if y==0):,}  fake={sum(1 for y in y_val if y==1):,})")
print(f"  Test rows : {len(X_test):,}  "
      f"(genuine={sum(1 for y in y_test if y==0):,}  fake={sum(1 for y in y_test if y==1):,})\n")


# ---------------------------------------------------------------------------
# Leakage check — focal/contrastive training data must contain neither the
# balanced TEST texts (evaluated on) nor the balanced VAL texts (tuned on).
# (Both use yelp_split_imbalanced.csv, so one check covers all ablation models.)
# ---------------------------------------------------------------------------
print("=" * 60)
print("Leakage check")
print("=" * 60)

if not os.path.exists(IMBALANCED_CSV):
    print(f"  WARNING: {IMBALANCED_CSV} not found — skipping leakage check.")
    print("  Generate it by running prepare_yelp_split_imbalanced.ipynb on Colab.")
    print()
else:
    test_norms = set(_norm(t) for t in X_test)
    val_norms  = set(_norm(t) for t in X_val)

    imb_df      = pd.read_csv(IMBALANCED_CSV)
    train_texts = imb_df[imb_df["split"] == "train"]["text"].tolist()
    train_norms = set(_norm(t) for t in train_texts)

    test_overlap = test_norms & train_norms
    val_overlap  = val_norms  & train_norms

    print(f"  Ablation training texts          : {len(train_norms):,}")
    print(f"  Balanced TEST texts in train     : {len(test_overlap):,}  (must be 0)")
    print(f"  Balanced VAL  texts in train     : {len(val_overlap):,}  (must be 0)")

    if test_overlap or val_overlap:
        print("\n  FAIL — balanced val/test reviews appear in ablation training data.")
        print("  Regenerate yelp_split_imbalanced.csv using prepare_yelp_split_imbalanced.ipynb.")
        sys.exit(1)
    print("\n  PASS — val + test are clean w.r.t. ablation training data.")
    print("  Re-tuning thresholds on the balanced val is valid for every model.\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tune_threshold(probs, y_true):
    """Threshold that maximises macro F1 on the (balanced) validation set."""
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(y_true, (probs >= t).astype(int), average="macro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(t)
    return round(best_t, 2), best_f


def row(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    fp  = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    fr  = recall_score   (y_true, y_pred, pos_label=1, zero_division=0)
    ff1 = f1_score       (y_true, y_pred, pos_label=1, zero_division=0)
    gp  = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    gr  = recall_score   (y_true, y_pred, pos_label=0, zero_division=0)
    gf1 = f1_score       (y_true, y_pred, pos_label=0, zero_division=0)

    # Confusion matrix (rows = actual, cols = predicted); labels fixed so it is
    # always 2x2 even if a model never predicts one class.
    tn, fpc, fnc, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fnc) if (tp + fnc) else 0.0   # fake recall      (prior-independent)
    tnr = tn / (tn + fpc) if (tn + fpc) else 0.0   # genuine recall   (prior-independent)
    fpr = 1.0 - tnr                                # false-positive rate

    return {
        "Model":      name,
        "Macro F1":   f"{mf1:.1%}",
        "Fake P":     f"{fp:.1%}",
        "Fake R":     f"{fr:.1%}",
        "Fake F1":    f"{ff1:.1%}",
        "Genuine P":  f"{gp:.1%}",
        "Genuine R":  f"{gr:.1%}",
        "Genuine F1": f"{gf1:.1%}",
        "_macro_f1":  mf1,
        "_tn": int(tn), "_fp": int(fpc), "_fn": int(fnc), "_tp": int(tp),
        "_tpr": tpr, "_tnr": tnr, "_fpr": fpr,
    }


def assemble(label, probs_val, probs_test, stored_thr):
    """Re-tune threshold on the balanced val, score the balanced test, and add
    threshold-free PR-AUC / ROC-AUC (fake = positive class)."""
    probs_val  = np.asarray(probs_val,  dtype=float)
    probs_test = np.asarray(probs_test, dtype=float)
    yv, yt     = np.asarray(y_val), np.asarray(y_test)

    thr, _ = tune_threshold(probs_val, yv)          # calibrated on balanced 50/50 val
    y_pred = (probs_test >= thr).astype(int)
    r = row(label, yt, y_pred)

    pr  = average_precision_score(yt, probs_test)   # PR-AUC  (threshold-free)
    roc = roc_auc_score(yt, probs_test)             # ROC-AUC (threshold-free)
    r["Thr*"]    = f"{thr:.2f}"
    r["PR-AUC"]  = f"{pr:.1%}"
    r["ROC-AUC"] = f"{roc:.1%}"
    r["_pr_auc"]  = pr
    r["_roc_auc"] = roc
    r["_thr"]     = thr
    r["_stored_thr"] = stored_thr
    return r


def project_precision(tpr, fpr, prior):
    """Fake-class precision a model would achieve at a given fake prior.
    Uses prior-independent TPR/FPR measured on the balanced test set, assuming
    the class-conditional feature distributions are the same in deployment."""
    denom = prior * tpr + (1.0 - prior) * fpr
    return (prior * tpr / denom) if denom > 0 else 0.0


def encode_lstm(texts, tok):
    seqs = tok.texts_to_sequences(texts)
    padded = []
    for s in seqs:
        s = s[:LSTM_MAX_LEN]
        s += [0] * (LSTM_MAX_LEN - len(s))
        padded.append(s)
    return np.array(padded, dtype=np.float32)


def run_lstm(model_dir, label):
    """Return (probs_val, probs_test, stored_threshold) or None if missing."""
    base = os.path.basename(model_dir).replace("_onnx", "")
    onnx_path   = os.path.join(model_dir, base + ".onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(model_dir, "model.onnx")
    tok_path    = os.path.join(model_dir, base + "_tokenizer.json")
    if not os.path.exists(tok_path):
        for fn in os.listdir(model_dir):
            if fn.endswith("_tokenizer.json"):
                tok_path = os.path.join(model_dir, fn)
                break
    thresh_path = os.path.join(model_dir, base + "_threshold.json")
    if not os.path.exists(thresh_path):
        thresh_path = os.path.join(model_dir, "threshold.json")

    if not os.path.exists(onnx_path):
        print(f"  [{label}] MISSING ({onnx_path}) — skipping")
        return None

    import onnxruntime as ort
    from score_reviews import _KerasTokenizer

    lstm_tok   = _KerasTokenizer.from_json(tok_path)
    stored_thr = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            stored_thr = json.load(f).get("threshold", 0.5)

    sess       = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    def predict(texts):
        out_probs = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = encode_lstm(texts[i:i+BATCH_SIZE], lstm_tok)
            out   = sess.run(None, {input_name: batch})[0].flatten()
            out_probs.extend(out.tolist())
        return out_probs

    return predict(X_val), predict(X_test), stored_thr


def run_distilbert(model_dir, label):
    """Return (probs_val, probs_test, stored_threshold) or None if missing."""
    if not os.path.isdir(model_dir):
        print(f"  [{label}] MISSING ({model_dir}) — skipping")
        return None

    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import pipeline as hf_pipeline

    thresh_path = os.path.join(model_dir, "threshold.json")
    stored_thr  = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            stored_thr = json.load(f).get("threshold", 0.5)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir, provider="CPUExecutionProvider"
    )
    bert_pipe = hf_pipeline(
        "text-classification", model=ort_model, tokenizer=tokenizer,
        device=-1, batch_size=8, truncation=True, max_length=BERT_MAX_LEN,
    )

    def predict(texts):
        out_probs = []
        for i in range(0, len(texts), BATCH_SIZE):
            for res in bert_pipe(texts[i:i+BATCH_SIZE]):
                score = res["score"]
                out_probs.append(score if res["label"] in ("LABEL_1", "fake", "1") else 1 - score)
        return out_probs

    return predict(X_val), predict(X_test), stored_thr


# ---------------------------------------------------------------------------
# Run all six models
# ---------------------------------------------------------------------------
rows = []


def evaluate(runner, model_dir, label):
    res = runner(model_dir, label)
    if res is None:
        return
    probs_val, probs_test, stored_thr = res
    r = assemble(label, probs_val, probs_test, stored_thr)
    rows.append(r)
    print(f"  [{label}] re-tuned thr={r['Thr*']} (shipped {stored_thr:.2f})  "
          f"Macro F1={r['Macro F1']}  PR-AUC={r['PR-AUC']}")


print("=" * 60)
print("Evaluating LSTM models...")
print("=" * 60)
for model_dir, label in [
    (LSTM_BASE_DIR,        "LSTM (baseline)"),
    (LSTM_FOCAL_DIR,       "LSTM (focal γ=2)"),
    (LSTM_CONTRASTIVE_DIR, "LSTM (contrastive)"),
]:
    evaluate(run_lstm, model_dir, label)

print()
print("=" * 60)
print("Evaluating DistilBERT models...")
print("=" * 60)
for model_dir, label in [
    (BERT_BASE_DIR,        "DistilBERT (baseline)"),
    (BERT_FOCAL_DIR,       "DistilBERT (focal γ=2)"),
    (BERT_CONTRASTIVE_DIR, "DistilBERT (contrastive)"),
]:
    evaluate(run_distilbert, model_dir, label)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
DISPLAY_COLS = ["Thr*", "Macro F1", "Fake P", "Fake R", "Fake F1",
                "Genuine P", "Genuine R", "Genuine F1", "PR-AUC", "ROC-AUC"]

table_w = 28 + 11 * len(DISPLAY_COLS)
print("\n\n" + "=" * table_w)
print("YELP FAKE DETECTION ABLATION — Baseline vs Focal vs Contrastive")
print("Test Set  (n={:,}  genuine={:,}  fake={:,})  |  threshold re-tuned on balanced val".format(
    len(y_test),
    sum(1 for y in y_test if y == 0),
    sum(1 for y in y_test if y == 1),
))
print("=" * table_w)

if not rows:
    print("  No models evaluated.")
else:
    headers = ["Model"] + DISPLAY_COLS
    widths  = [28] + [11] * len(DISPLAY_COLS)
    print("".join(h.rjust(w) for h, w in zip(headers, widths)))
    print("-" * sum(widths))
    for _, r in pd.DataFrame(rows).sort_values("_macro_f1", ascending=False).iterrows():
        vals = [r["Model"]] + [r[c] for c in DISPLAY_COLS]
        print("".join(str(v).rjust(w) for v, w in zip(vals, widths)))

    print("=" * table_w)
    print("Thr*    = decision threshold RE-TUNED on the balanced val (fair across models).")
    print("Macro F1 / Fake / Genuine = at Thr* on the balanced test.")
    print("PR-AUC / ROC-AUC = THRESHOLD-FREE ranking quality (fake = positive). The cleanest")
    print("                   'is it a good detector' numbers — independent of any threshold.")

    # Delta summary — macro F1 (threshold-dependent) and PR-AUC (threshold-free)
    by_name = {r["Model"]: r for r in rows}
    print()
    print("Delta summary (variant − baseline):")
    print(f"  {'':<28}{'ΔMacro F1':>12}{'ΔPR-AUC':>12}")
    pairs = [
        ("LSTM (baseline)",       "LSTM (focal γ=2)",         "LSTM focal"),
        ("LSTM (baseline)",       "LSTM (contrastive)",       "LSTM contrastive"),
        ("DistilBERT (baseline)", "DistilBERT (focal γ=2)",   "DistilBERT focal"),
        ("DistilBERT (baseline)", "DistilBERT (contrastive)", "DistilBERT contrastive"),
    ]
    for base_name, var_name, display in pairs:
        if base_name in by_name and var_name in by_name:
            dmf1 = by_name[var_name]["_macro_f1"] - by_name[base_name]["_macro_f1"]
            dpr  = by_name[var_name]["_pr_auc"]   - by_name[base_name]["_pr_auc"]
            print(f"  {display:<28}{('+' if dmf1>=0 else '')+format(dmf1,'.1%'):>12}"
                  f"{('+' if dpr>=0 else '')+format(dpr,'.1%'):>12}")
    print()

    # -----------------------------------------------------------------------
    # Confusion matrices — what each model actually gets right/wrong per class
    # -----------------------------------------------------------------------
    print("=" * table_w)
    print("CONFUSION MATRICES  (balanced test, at Thr* — rows = actual, cols = predicted)")
    print("=" * table_w)
    for r in sorted(rows, key=lambda x: x["_macro_f1"], reverse=True):
        print(f"\n  {r['Model']}  (Thr* = {r['Thr*']})")
        print(f"                    pred GENUINE   pred FAKE")
        print(f"    actual GENUINE   {r['_tn']:>11,}   {r['_fp']:>9,}")
        print(f"    actual FAKE      {r['_fn']:>11,}   {r['_tp']:>9,}")
        print(f"    Fake recall (caught): {r['_tpr']:.1%}   |   "
              f"False-positive rate: {r['_fpr']:.1%}   |   "
              f"Genuine recall: {r['_tnr']:.1%}")

    # -----------------------------------------------------------------------
    # Projected deployment at the real-world fake prior
    # -----------------------------------------------------------------------
    pi   = REAL_WORLD_FAKE_RATE
    n_1k = 1000
    n_fake_1k    = int(round(n_1k * pi))
    n_genuine_1k = n_1k - n_fake_1k

    print("\n\n" + "=" * table_w)
    print(f"PROJECTED DEPLOYMENT AT REAL-WORLD PRIOR (~{pi:.0%} fake)")
    print("=" * table_w)
    print("The test set is balanced 50/50, but recall and false-positive rate do not depend")
    print("on the class prior — so we re-project fake-class PRECISION to the real ~13% fake rate.")
    print("This is the honest 'is it a usable detector' view: at 13% fake, false positives dominate.")
    print(f"\nFraming per {n_1k:,} live reviews:  {n_fake_1k} fake  /  {n_genuine_1k} genuine\n")

    hdr = ["Model", "Fake P@13%", "Fake R", "Fake F1@13%", "Flagged/1k", "TrueFake/1k", "FalseFlag/1k"]
    w   = [28, 12, 10, 13, 12, 13, 14]
    print("".join(h.rjust(x) for h, x in zip(hdr, w)))
    print("-" * sum(w))

    proj_rows = []
    for r in rows:
        p_at  = project_precision(r["_tpr"], r["_fpr"], pi)
        r_at  = r["_tpr"]                                  # recall unchanged by prior
        f1_at = (2 * p_at * r_at / (p_at + r_at)) if (p_at + r_at) else 0.0
        tp_1k = n_fake_1k    * r["_tpr"]
        fp_1k = n_genuine_1k * r["_fpr"]
        proj_rows.append((r["Model"], p_at, r_at, f1_at, tp_1k + fp_1k, tp_1k, fp_1k))

    for m, p_at, r_at, f1_at, flagged, tp_1k, fp_1k in sorted(proj_rows, key=lambda t: t[3], reverse=True):
        vals = [m, f"{p_at:.1%}", f"{r_at:.1%}", f"{f1_at:.1%}",
                f"{flagged:.0f}", f"{tp_1k:.0f}", f"{fp_1k:.0f}"]
        print("".join(v.rjust(x) for v, x in zip(vals, w)))

    print("=" * table_w)
    print("Fake P@13% = precision among flagged reviews when only 13% are truly fake.")
    print("Flagged/1k = reviews the model flags as fake per 1,000;  of those, TrueFake/1k are")
    print("really fake and FalseFlag/1k are genuine reviews wrongly flagged.")
    print("A strong balanced macro F1 with low Fake P@13% means the model is not deployment-ready")
    print("as a standalone text detector — corroborate with behavioural signals before flagging.")
    print()

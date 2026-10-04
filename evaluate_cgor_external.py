"""
evaluate_cgor_external.py — CROSS-DATASET (out-of-distribution) generalization test
for the BROADENED CG/OR (machine-generated vs human) fake-review detector.

Per the thesis advisor's suggestion ("generalize your solution on a different
dataset and test its behaviour"), this runs the models trained on
data/fake_reviews_broadened.csv ZERO-SHOT (no retraining) on an INDEPENDENT
public corpus they have never seen:

    HC3  (Hello-SimpleAI/HC3) — Human vs ChatGPT answers across 5 domains
         (reddit_eli5, open_qa, wiki_csai, medicine, finance).

Why HC3 is a strong probe: it differs from our training data in BOTH
  (1) the generator  — ChatGPT, never used to train these detectors, and
  (2) the text TYPE  — Q&A answers, not product/movie reviews.
So good performance here means the detector learned a transferable
machine-vs-human signal rather than a review-specific or generator-specific
fingerprint. A drop is itself a reportable finding (domain + generator shift).

Label convention (matches training): 1 = machine/CG/fake, 0 = human/OR/genuine.

Models evaluated (whichever are present on disk):
  * 8 sklearn pipelines  -> data/fake_broadened_sklearn_models/*.joblib (+ thresholds.json)
  * LSTM (ONNX)          -> data/fake_broadened_lstm.onnx (+ _tokenizer.json, _threshold.json)
  * DistilBERT (ONNX)    -> data/fake_broadened_distilbert_onnx/ (+ threshold.json)

The DL models come from Colab; if their files aren't downloaded yet the script
just skips them and evaluates whatever is available.

RUN
    python evaluate_cgor_external.py
    python evaluate_cgor_external.py --per-class 5000     # cap reviews per class
"""

import os
import sys
import json
import argparse

import numpy as np

from download_negatives import clean   # same text cleaning used to build the dataset

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
SK_DIR      = os.path.join(DATA_DIR, "fake_broadened_sklearn_models")
LSTM_DIR    = os.path.join(DATA_DIR, "fake_broadened_lstm_onnx")
LSTM_ONNX   = os.path.join(LSTM_DIR, "fake_broadened_lstm.onnx")
LSTM_TOK    = os.path.join(LSTM_DIR, "fake_broadened_lstm_tokenizer.json")
LSTM_THR    = os.path.join(LSTM_DIR, "fake_broadened_lstm_threshold.json")
DB_DIR      = os.path.join(DATA_DIR, "fake_broadened_distilbert_onnx")
LSTM_MAX_LEN = 250
DB_MAX_LEN   = 256
MIN_WORDS    = 5     # matches download_negatives.clean() minimum


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# 1) Load HC3 (independent corpus) -> balanced (text, label) test set
# --------------------------------------------------------------------------- #
def load_hc3(per_class):
    """Return (texts, labels) balanced 1=machine(ChatGPT) / 0=human, cleaned + deduped."""
    rows = None
    # datasets 4.x dropped script loaders; pull the raw all.jsonl directly + cache it.
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="Hello-SimpleAI/HC3", filename="all.jsonl",
                               repo_type="dataset")
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        log(f"HC3: loaded {len(rows)} question rows from {os.path.basename(path)}")
    except Exception as e:
        log(f"hf_hub_download failed ({e}); trying load_dataset ...")
        from datasets import load_dataset
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
        rows = list(ds)
        log(f"HC3: loaded {len(rows)} question rows via load_dataset")

    human, machine, seen = [], [], set()

    def take(text, bucket):
        t = clean(text)
        if not t:
            return
        k = t.lower()
        if k in seen:
            return
        seen.add(k)
        bucket.append(t)

    for r in rows:
        for h in (r.get("human_answers") or []):
            take(h, human)
        for c in (r.get("chatgpt_answers") or []):
            take(c, machine)

    log(f"HC3 cleaned: human={len(human)}  machine={len(machine)} (min {MIN_WORDS} words, deduped)")
    rng = np.random.RandomState(42)
    n = min(len(human), len(machine), per_class)
    human   = list(rng.choice(human,   size=n, replace=False))
    machine = list(rng.choice(machine, size=n, replace=False))
    texts  = human + machine
    labels = np.array([0] * n + [1] * n)
    log(f"HC3 balanced test set: {len(texts)} texts ({n} human + {n} machine)")
    return texts, labels


# --------------------------------------------------------------------------- #
# 2) Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    return {
        "acc":      accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "machine_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),  # caught AI
        "human_specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),  # human kept
    }


# --------------------------------------------------------------------------- #
# 3) sklearn models
# --------------------------------------------------------------------------- #
def eval_sklearn(texts, y_true, results, pred_matrix):
    if not os.path.isdir(SK_DIR):
        log(f"(skipping sklearn — {SK_DIR} not found)")
        return
    import joblib
    thr_path = os.path.join(SK_DIR, "thresholds.json")
    thresholds = json.load(open(thr_path)) if os.path.exists(thr_path) else {}
    for name in ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]:
        p = os.path.join(SK_DIR, f"{name}.joblib")
        if not os.path.exists(p):
            continue
        try:
            pipe = joblib.load(p)
            proba = pipe.predict_proba(texts)[:, 1]
            thr = thresholds.get(name, 0.5)
            pred = (proba >= thr).astype(int)
            results[f"sk_{name}"] = metrics(y_true, pred)
            pred_matrix.append(pred)
        except Exception as e:
            log(f"  sklearn {name} failed: {e}")


# --------------------------------------------------------------------------- #
# 4) LSTM ONNX  (pure-Python Keras tokenizer, no TensorFlow)
# --------------------------------------------------------------------------- #
class _KerasTokenizer:
    """Minimal keras Tokenizer reimplementation for inference (mirrors score_reviews.py)."""
    def __init__(self, word_index, filters, lower, num_words):
        self.word_index = word_index
        self._filters   = str.maketrans("", "", filters)
        self.lower      = lower
        self.num_words  = num_words

    @classmethod
    def from_json(cls, path):
        cfg = json.load(open(path, encoding="utf-8"))["config"]
        wi = cfg["word_index"]
        if isinstance(wi, str):
            wi = json.loads(wi)
        return cls(wi, cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
                   cfg.get("lower", True), cfg.get("num_words"))

    def texts_to_sequences(self, texts):
        seqs = []
        for text in texts:
            if self.lower:
                text = text.lower()
            text = text.translate(self._filters)
            seq = []
            for w in text.split():
                idx = self.word_index.get(w)
                if idx is not None and (self.num_words is None or idx < self.num_words):
                    seq.append(idx)
            seqs.append(seq)
        return seqs


def eval_lstm(texts, y_true, results, pred_matrix):
    if not (os.path.exists(LSTM_ONNX) and os.path.exists(LSTM_TOK)):
        log("(skipping LSTM — broadened ONNX/tokenizer not found; download from Colab)")
        return
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(LSTM_ONNX, providers=["CPUExecutionProvider"])
        tok = _KerasTokenizer.from_json(LSTM_TOK)
        thr = json.load(open(LSTM_THR)).get("threshold", 0.5) if os.path.exists(LSTM_THR) else 0.5
        in_name = sess.get_inputs()[0].name
        seqs = tok.texts_to_sequences(texts)
        proba = np.zeros(len(texts), dtype=np.float32)
        B = 512
        for i in range(0, len(texts), B):
            batch = seqs[i:i + B]
            arr = np.zeros((len(batch), LSTM_MAX_LEN), dtype=np.float32)
            for j, s in enumerate(batch):
                s = s[:LSTM_MAX_LEN]
                arr[j, :len(s)] = s
            out = sess.run(None, {in_name: arr})[0].reshape(-1)
            proba[i:i + len(batch)] = out
        pred = (proba >= thr).astype(int)
        results["lstm"] = metrics(y_true, pred)
        pred_matrix.append(pred)
    except Exception as e:
        log(f"  LSTM failed: {e}")


# --------------------------------------------------------------------------- #
# 5) DistilBERT ONNX (optimum)
# --------------------------------------------------------------------------- #
def eval_distilbert(texts, y_true, results, pred_matrix):
    if not os.path.isdir(DB_DIR):
        log("(skipping DistilBERT — broadened ONNX dir not found; download from Colab)")
        return
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        mdl = ORTModelForSequenceClassification.from_pretrained(DB_DIR, provider="CPUExecutionProvider")
        tok = AutoTokenizer.from_pretrained(DB_DIR)
        thr_path = os.path.join(DB_DIR, "threshold.json")
        thr = json.load(open(thr_path)).get("threshold", 0.5) if os.path.exists(thr_path) else 0.5
        proba = np.zeros(len(texts), dtype=np.float32)
        B = 64
        for i in range(0, len(texts), B):
            enc = tok(texts[i:i + B], truncation=True, padding=True,
                      max_length=DB_MAX_LEN, return_tensors="np")
            logits = mdl(**enc).logits
            logits = np.asarray(logits, dtype=np.float64)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            proba[i:i + logits.shape[0]] = (e / e.sum(axis=1, keepdims=True))[:, 1]
        pred = (proba >= thr).astype(int)
        results["distilbert"] = metrics(y_true, pred)
        pred_matrix.append(pred)
    except Exception as e:
        log(f"  DistilBERT failed: {e}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=8000,
                    help="max reviews per class drawn from HC3 (default 8000)")
    args = ap.parse_args()

    log("=" * 72)
    log("CROSS-DATASET GENERALIZATION — broadened CG/OR detector on HC3 (zero-shot)")
    log("=" * 72)
    texts, y_true = load_hc3(args.per_class)

    results, pred_matrix = {}, []
    log("\nrunning sklearn ...");     eval_sklearn(texts, y_true, results, pred_matrix)
    log("running LSTM ...");          eval_lstm(texts, y_true, results, pred_matrix)
    log("running DistilBERT ...");    eval_distilbert(texts, y_true, results, pred_matrix)

    if pred_matrix:
        maj = (np.mean(pred_matrix, axis=0) >= 0.5).astype(int)
        results["ENSEMBLE_majority"] = metrics(y_true, maj)

    log("\n" + "=" * 72)
    log(f"HC3 ZERO-SHOT RESULTS  ({len(texts)} texts, balanced)  1=machine 0=human")
    log("=" * 72)
    log(f"{'model':<22}{'acc':>9}{'macroF1':>10}{'machineRecall':>15}{'humanSpec':>12}")
    log("-" * 72)
    for name, m in results.items():
        log(f"{name:<22}{m['acc']:>8.1%} {m['macro_f1']:>9.1%} "
            f"{m['machine_recall']:>14.1%} {m['human_specificity']:>11.1%}")
    log("-" * 72)
    log("machineRecall = fraction of ChatGPT answers caught as machine (the OOD generalization metric);")
    log("humanSpec     = fraction of human answers correctly kept (false-positive control).")
    log("Compare to in-distribution test acc (~0.96) to read the generalization gap.")


if __name__ == "__main__":
    main()

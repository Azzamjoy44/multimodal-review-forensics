"""
model_a.py — the GENERAL all-domain CG/OR detector ("Model A") for the Dashboard
upload box (/analyze, Tier 1).

Scores arbitrary uploaded review text with the broadened multi-domain ensemble
(8 sklearn + LSTM ONNX + DistilBERT ONNX, trained on products/movies/restaurants/
apps/books/games × many generator families). 1 = fake (machine/CG), 0 = genuine
(human/OR). Output per text is the same `fake_models` shape the frontend already
renders: {name: {label, confidence, reasons}} for knn/rf/dt/lr/nb/xgb/adaboost/mlp
+ lstm + distilbert.

Reuses score_reviews' phrase extractors (reasons → highlight phrases) so Analyze
highlighting matches the browse cards exactly. Models:
    data/fake_broadened_sklearn_models/*.joblib (+ thresholds.json)
    data/fake_broadened_lstm_onnx/   (onnx + keras tokenizer json + threshold json)
    data/fake_broadened_distilbert_onnx/ (model.onnx + HF tokenizer + threshold.json)
"""

import os
import json
import numpy as np

# reuse the browse-card phrase extractors so Analyze highlighting matches exactly
# (data-driven from per-model reasons; safe to import — main.py loads score_reviews first)
from score_reviews import (_extract_reasons, _get_suspicious_phrases,
                           _get_genuine_phrases, _split_sentences)

_DATA   = os.path.join(os.path.dirname(__file__), "data")
SK_DIR  = os.path.join(_DATA, "fake_broadened_sklearn_models")
LSTM_DIR = os.path.join(_DATA, "fake_broadened_lstm_onnx")
DB_DIR  = os.path.join(_DATA, "fake_broadened_distilbert_onnx")
SK_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
LSTM_MAX_LEN = 250
DB_MAX_LEN   = 256

# ---------------------------------------------------------------------------
# Minimal pure-Python Keras tokenizer (mirrors score_reviews._KerasTokenizer)
# ---------------------------------------------------------------------------
class _KerasTokenizer:
    def __init__(self, word_index, filters, lower, num_words):
        self.word_index = word_index
        self._filters = str.maketrans("", "", filters)
        self.lower = lower
        self.num_words = num_words

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


def _pad(seqs, maxlen):
    out = np.zeros((len(seqs), maxlen), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s[:maxlen]
        out[i, :len(s)] = s
    return out


# ---------------------------------------------------------------------------
# Lazy-loaded model caches
# ---------------------------------------------------------------------------
_sk = None; _sk_thr = None
_lstm = None
_db = None


def _load_sklearn():
    global _sk, _sk_thr
    if _sk is not None:
        return _sk, _sk_thr
    import joblib
    _sk, _sk_thr = {}, {}
    thr_path = os.path.join(SK_DIR, "thresholds.json")
    thresholds = json.load(open(thr_path)) if os.path.exists(thr_path) else {}
    for n in SK_NAMES:
        p = os.path.join(SK_DIR, f"{n}.joblib")
        if os.path.exists(p):
            try:
                _sk[n] = joblib.load(p); _sk_thr[n] = thresholds.get(n, 0.5)
            except Exception as e:
                print(f"Model A: sklearn {n} load failed — {e}")
    print(f"Model A: loaded {len(_sk)} broadened sklearn models.")
    return _sk, _sk_thr


def _load_lstm():
    global _lstm
    if _lstm is not None:
        return _lstm
    onnx = os.path.join(LSTM_DIR, "fake_broadened_lstm.onnx")
    tok  = os.path.join(LSTM_DIR, "fake_broadened_lstm_tokenizer.json")
    thrp = os.path.join(LSTM_DIR, "fake_broadened_lstm_threshold.json")
    if os.path.exists(onnx) and os.path.exists(tok):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
            thr = json.load(open(thrp)).get("threshold", 0.5) if os.path.exists(thrp) else 0.5
            _lstm = (sess, _KerasTokenizer.from_json(tok), thr, sess.get_inputs()[0].name)
            print("Model A: broadened LSTM ONNX loaded.")
        except Exception as e:
            print(f"Model A: LSTM load failed — {e}"); _lstm = False
    else:
        _lstm = False
    return _lstm


def _load_db():
    global _db
    if _db is not None:
        return _db
    if os.path.isdir(DB_DIR) and os.path.exists(os.path.join(DB_DIR, "model.onnx")):
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            sess = ort.InferenceSession(os.path.join(DB_DIR, "model.onnx"), providers=["CPUExecutionProvider"])
            tok = AutoTokenizer.from_pretrained(DB_DIR)
            thrp = os.path.join(DB_DIR, "threshold.json")
            thr = json.load(open(thrp)).get("threshold", 0.5) if os.path.exists(thrp) else 0.5
            _db = (sess, tok, thr)
            print("Model A: broadened DistilBERT ONNX loaded.")
        except Exception as e:
            print(f"Model A: DistilBERT load failed — {e}"); _db = False
    else:
        _db = False
    return _db


def preload():
    """Eager-load all Model-A models (call at server startup)."""
    _load_sklearn(); _load_lstm(); _load_db()


# ---------------------------------------------------------------------------
def _conf(p, label):
    return round(float(p if label == 1 else 1 - p) * 100)


def _score_sklearn(texts):
    sk, thr = _load_sklearn()
    res = [{} for _ in texts]
    for n, pipe in sk.items():
        try:
            proba = pipe.predict_proba(texts)[:, 1]
            t = thr[n]
            fn = pipe.named_steps["tfidf"].get_feature_names_out()
            vecs = pipe.named_steps["tfidf"].transform(texts)
            for i in range(len(texts)):
                label = int(proba[i] >= t)
                # class-specific word attribution (same extractor the browse cards use)
                reasons = _extract_reasons(pipe, vecs[i], fn, label, ("genuine", "fake"))
                res[i][n] = {"label": label, "confidence": _conf(proba[i], label), "reasons": reasons}
        except Exception as e:
            for r in res:
                r[n] = {"label": None, "confidence": None, "reasons": [f"error: {e}"]}
    return res


def _score_lstm(texts):
    l = _load_lstm()
    if not l:
        return [{"label": None, "confidence": None, "reasons": ["LSTM unavailable"]} for _ in texts]
    sess, tok, thr, in_name = l
    arr = _pad(tok.texts_to_sequences(texts), LSTM_MAX_LEN)
    proba = sess.run(None, {in_name: arr})[0].reshape(-1)
    out = []
    for p in proba:
        label = int(p >= thr)
        out.append({"label": label, "confidence": _conf(p, label),
                    "reasons": ["sequence-level machine-text signal (BiLSTM)"]})
    return out


def _score_db(texts):
    d = _load_db()
    if not d:
        return [{"label": None, "confidence": None, "reasons": ["DistilBERT unavailable"]} for _ in texts]
    sess, tok, thr = d
    out = []
    B = 32
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i + B], truncation=True, padding="max_length",
                  max_length=DB_MAX_LEN, return_tensors="np")
        feed = {x.name: enc[x.name].astype("int64") for x in sess.get_inputs() if x.name in enc}
        logits = np.asarray(sess.run(None, feed)[0], dtype=np.float64)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        p1 = (e / e.sum(axis=1, keepdims=True))[:, 1]
        for p in p1:
            label = int(p >= thr)
            out.append({"label": label, "confidence": _conf(p, label),
                        "reasons": ["contextual machine-text representation (DistilBERT)"]})
    return out


def _dl_sentences(text):
    """Per-sentence verdicts from the broadened LSTM + DistilBERT, so the frontend
    can sentence-highlight when an LSTM/DistilBERT dot is clicked. Returns
    (suspicious, genuine) lists of {sentence, models:[...]} — each sentence lands in
    one list per DL model depending on whether that model scores it >= its threshold."""
    sents = _split_sentences(text)
    if not sents:
        return [], []
    susp, genu = {}, {}   # sentence -> set(model display names)

    l = _load_lstm()
    if l:
        try:
            sess, tok, thr, in_name = l
            arr = _pad(tok.texts_to_sequences(sents), LSTM_MAX_LEN)
            p = sess.run(None, {in_name: arr})[0].reshape(-1)
            for i, prob in enumerate(p):
                (susp if prob >= thr else genu).setdefault(sents[i], set()).add("LSTM")
        except Exception:
            pass

    d = _load_db()
    if d:
        try:
            sess, tok, thr = d
            enc = tok(sents, truncation=True, padding="max_length",
                      max_length=DB_MAX_LEN, return_tensors="np")
            feed = {x.name: enc[x.name].astype("int64") for x in sess.get_inputs() if x.name in enc}
            logits = np.asarray(sess.run(None, feed)[0], dtype=np.float64)
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            p1 = (e / e.sum(axis=1, keepdims=True))[:, 1]
            for i, prob in enumerate(p1):
                (susp if prob >= thr else genu).setdefault(sents[i], set()).add("DistilBERT")
        except Exception:
            pass

    susp_l = [{"sentence": s, "models": sorted(m)} for s, m in susp.items()]
    genu_l = [{"sentence": s, "models": sorted(m), "type": "genuine"} for s, m in genu.items()]
    return susp_l, genu_l


def score_fake_distilbert(texts):
    """DistilBERT-only fake verdicts (broadened all-domain detector) — used by the
    Dashboard user-history chart, which only needs the single strongest model, not
    the full 10-model ensemble. Returns [{label, confidence, reasons}] per text."""
    texts = [("" if t is None else str(t)) for t in texts]
    if not texts:
        return []
    return _score_db(texts)


def score_fake(texts):
    """Return list (per text) of a rich dict the frontend can highlight:
        {fake_models, suspicious_phrases, genuine_phrases,
         dl_suspicious_sentences, dl_genuine_sentences}
    fake_models = 8 sklearn + lstm + distilbert ({label, confidence, reasons})."""
    texts = [("" if t is None else str(t)) for t in texts]
    if not texts:
        return []
    sk = _score_sklearn(texts)
    ls = _score_lstm(texts)
    db = _score_db(texts)
    out = []
    for i in range(len(texts)):
        fm = dict(sk[i]); fm["lstm"] = ls[i]; fm["distilbert"] = db[i]
        dl_s, dl_g = _dl_sentences(texts[i])
        out.append({
            "fake_models":             fm,
            "suspicious_phrases":      _get_suspicious_phrases(texts[i], fm),
            "genuine_phrases":         _get_genuine_phrases(texts[i], fm),
            "dl_suspicious_sentences": dl_s,
            "dl_genuine_sentences":    dl_g,
        })
    return out

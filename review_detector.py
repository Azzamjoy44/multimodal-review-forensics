"""
review_detector.py — the REVIEW vs NON-REVIEW gate used by the Dashboard
"Analyze" feature.

Loads the review-detector models trained on data/review_detection.csv:
    * 8 sklearn pipelines  -> data/review_detector_sklearn_models/<name>.joblib (+ thresholds.json)
    * LSTM ONNX            -> data/review_detector_lstm_onnx/
    * DistilBERT ONNX      -> data/review_detector_distilbert_onnx/

`classify_texts(texts)` returns, per text, every model's vote (1 = review,
0 = non-review) at that model's tuned threshold, plus the majority decision and
the list of models that dissented. Missing model files are skipped gracefully,
so the gate works as soon as ANY model is present.
"""

import os
import json
import re
import threading

import numpy as np

# reuse the proven word-attribution + sentence splitter (main.py loads score_reviews first)
from score_reviews import _extract_reasons, _split_sentences

_DISPLAY = {"knn": "KNN", "rf": "Random Forest", "dt": "Decision Tree", "lr": "LR",
            "nb": "Naïve Bayes", "xgb": "XGBoost", "adaboost": "AdaBoost", "mlp": "MLP",
            "lstm": "LSTM", "distilbert": "DistilBERT"}

_DATA      = os.path.join(os.path.dirname(__file__), "data")
SK_DIR     = os.path.join(_DATA, "review_detector_sklearn_models")
LSTM_DIR   = os.path.join(_DATA, "review_detector_lstm_onnx")
BERT_DIR   = os.path.join(_DATA, "review_detector_distilbert_onnx")
LSTM_MAX_LEN = 250    # must match training (MAX_LEN in the LSTM notebook)
BERT_MAX_LEN = 256    # must match training (MAX_LEN in the DistilBERT notebook)
SK_NAMES   = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]

_lock     = threading.Lock()
_sk       = None     # {name: pipeline}
_sk_thr   = None     # {name: threshold}
_lstm     = None     # (session, tokenizer, threshold)  or  False
_bert     = None     # (session, tokenizer, threshold)  or  False


# ---------------------------------------------------------------------------
# Keras tokenizer (pure-python, mirrors score_reviews._KerasTokenizer)
# ---------------------------------------------------------------------------
class _KerasTokenizer:
    def __init__(self, word_index, filters, lower, num_words):
        self.word_index = word_index
        self._filters   = str.maketrans("", "", filters)
        self.lower      = lower
        self.num_words  = num_words

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)["config"]
        wi = cfg["word_index"]
        if isinstance(wi, str):
            wi = json.loads(wi)
        return cls(wi, cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
                   cfg.get("lower", True), cfg.get("num_words"))

    def texts_to_sequences(self, texts):
        out = []
        for text in texts:
            if self.lower:
                text = text.lower()
            text = text.translate(self._filters)
            seq = []
            for w in text.split():
                idx = self.word_index.get(w)
                if idx is not None and (self.num_words is None or idx < self.num_words):
                    seq.append(idx)
            out.append(seq)
        return out


def _lstm_pad(seqs, maxlen):
    out = np.zeros((len(seqs), maxlen), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = seq[:maxlen]
        out[i, :len(seq)] = seq
    return out


# ---------------------------------------------------------------------------
# lazy loaders
# ---------------------------------------------------------------------------
def _load_sklearn():
    global _sk, _sk_thr
    if _sk is not None:
        return
    import joblib
    sk, thr = {}, {}
    tp = os.path.join(SK_DIR, "thresholds.json")
    if os.path.exists(tp):
        with open(tp) as f:
            thr = json.load(f)
    for n in SK_NAMES:
        p = os.path.join(SK_DIR, f"{n}.joblib")
        if os.path.exists(p):
            try:
                sk[n] = joblib.load(p)
            except Exception as e:
                print(f"[review_detector] skip sklearn {n}: {e}")
    _sk, _sk_thr = sk, thr


def _load_lstm():
    global _lstm
    if _lstm is not None:
        return
    onnx = os.path.join(LSTM_DIR, "review_detector_lstm.onnx")
    if not os.path.exists(onnx):
        _lstm = False
        return
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        tok  = _KerasTokenizer.from_json(os.path.join(LSTM_DIR, "review_detector_lstm_tokenizer.json"))
        with open(os.path.join(LSTM_DIR, "review_detector_lstm_threshold.json")) as f:
            thr = json.load(f)["threshold"]
        _lstm = (sess, tok, thr)
    except Exception as e:
        print(f"[review_detector] LSTM load failed: {e}")
        _lstm = False


def _load_bert():
    global _bert
    if _bert is not None:
        return
    onnx = os.path.join(BERT_DIR, "model.onnx")
    if not os.path.exists(onnx):
        _bert = False
        return
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
        sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        tok  = AutoTokenizer.from_pretrained(BERT_DIR)
        tp = os.path.join(BERT_DIR, "threshold.json")
        thr = json.load(open(tp))["threshold"] if os.path.exists(tp) else 0.5
        _bert = (sess, tok, thr)
    except Exception as e:
        print(f"[review_detector] DistilBERT load failed: {e}")
        _bert = False


def load_all():
    """Pre-load every available review-detector model (call once at startup)."""
    with _lock:
        _load_sklearn()
        _load_lstm()
        _load_bert()


def available_models():
    load_all()
    names = list(_sk.keys())
    if _lstm:
        names.append("lstm")
    if _bert:
        names.append("distilbert")
    return names


# ---------------------------------------------------------------------------
# per-model P(review)
# ---------------------------------------------------------------------------
def _bert_probs(texts):
    sess, tok, _ = _bert
    enc = tok(list(texts), max_length=BERT_MAX_LEN, padding=True,
              truncation=True, return_tensors="np")
    in_names = [i.name for i in sess.get_inputs()]
    feed = {k: enc[k].astype(np.int64) for k in in_names if k in enc}
    logits = sess.run(None, feed)[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def _build_phrases(text, reasons_by_model, target_vote):
    """Words the sklearn review-detector models attributed to their vote, for
    highlighting. Stop words are KEPT (pronouns like 'I'/'my' are strong review
    signals — see module docstring). target_vote 1 = review, 0 = non-review."""
    text_lower = text.lower()
    pm = {}   # word_lower -> {phrase, models:set}
    for name, info in reasons_by_model.items():
        if info["vote"] != target_vote:
            continue
        for reason in info["reasons"]:
            m = re.search(r'"([^"]+)"', reason)
            if not m:
                continue
            w = m.group(1).strip()
            if w.lower() in text_lower:
                pm.setdefault(w.lower(), {"phrase": w, "models": set()})
                pm[w.lower()]["models"].add(_DISPLAY.get(name, name.upper()))
    kind = "review_signal" if target_vote == 1 else "nonreview_signal"
    return [{"phrase": v["phrase"], "type": kind, "models": sorted(v["models"])}
            for v in pm.values()]


def _dl_review_sentences(text):
    """Per-sentence verdicts from the review-detector LSTM + DistilBERT, so an
    LSTM/DistilBERT dot can sentence-highlight. Returns (review, non-review) lists
    of {sentence, models:[...], type}."""
    sents = _split_sentences(text)
    if not sents:
        return [], []
    rev, nonrev = {}, {}
    if _lstm:
        try:
            sess, tok, thr = _lstm
            probs = sess.run(None, {"input_layer": _lstm_pad(tok.texts_to_sequences(sents), LSTM_MAX_LEN)})[0].flatten()
            for i, p in enumerate(probs):
                (rev if p >= thr else nonrev).setdefault(sents[i], set()).add("LSTM")
        except Exception:
            pass
    if _bert:
        try:
            probs = _bert_probs(sents); thr = float(_bert[2])
            for i, p in enumerate(probs):
                (rev if p >= thr else nonrev).setdefault(sents[i], set()).add("DistilBERT")
        except Exception:
            pass
    return ([{"sentence": s, "models": sorted(m), "type": "review"} for s, m in rev.items()],
            [{"sentence": s, "models": sorted(m), "type": "nonreview"} for s, m in nonrev.items()])


# A text is gated as NON-REVIEW only if at least this many models vote non-review (out of ~10).
# Higher = more permissive toward "review" (fewer real reviews wrongly rejected by the gate).
NON_REVIEW_MIN_VOTES = 8


def classify_texts(texts):
    """Return one dict per input text:
        {is_review, review_votes, total_models, per_model{name:{vote,p_review}},
         dissenting_models, review_phrases, nonreview_phrases,
         dl_review_sentences, dl_nonreview_sentences}
    `dissenting_models` = models that voted NON-review. The phrase/sentence lists
    let the frontend highlight what made each model vote review vs non-review."""
    load_all()
    texts = [("" if t is None else str(t)) for t in texts]
    if not texts:
        return []

    per_model = {}   # name -> (probs ndarray, threshold)
    sk_extra  = {}   # name -> (tfidf matrix, feature_names)  for word attribution
    for name, pipe in _sk.items():
        try:
            per_model[name] = (pipe.predict_proba(texts)[:, 1], float(_sk_thr.get(name, 0.5)))
            try:
                vec = pipe.named_steps["tfidf"]
                sk_extra[name] = (vec.transform(texts), vec.get_feature_names_out())
            except Exception:
                sk_extra[name] = None
        except Exception as e:
            print(f"[review_detector] sklearn {name} predict failed: {e}")
    if _lstm:
        sess, tok, thr = _lstm
        probs = sess.run(None, {"input_layer": _lstm_pad(tok.texts_to_sequences(texts), LSTM_MAX_LEN)})[0].flatten()
        per_model["lstm"] = (probs, float(thr))
    if _bert:
        per_model["distilbert"] = (_bert_probs(texts), float(_bert[2]))

    model_names = list(per_model.keys())
    total = len(model_names)
    results = []
    for i in range(len(texts)):
        votes = {n: int(per_model[n][0][i] >= per_model[n][1]) for n in model_names}
        review_votes = sum(votes.values())
        # word attribution per sklearn model (class-specific, stop words kept)
        reasons_by_model = {}
        for name in _sk:
            if name in per_model and sk_extra.get(name):
                vecs, fn = sk_extra[name]
                try:
                    rs = _extract_reasons(_sk[name], vecs[i], fn, votes[name], ("non-review", "review"))
                except Exception:
                    rs = []
                reasons_by_model[name] = {"vote": votes[name], "reasons": rs}
        dl_rev, dl_nonrev = _dl_review_sentences(texts[i])
        results.append({
            "is_review":      total > 0 and (total - review_votes) < NON_REVIEW_MIN_VOTES,   # non-review only if >=8 models say non-review
            "review_votes":   review_votes,
            "total_models":   total,
            "per_model":      {n: {"vote": votes[n],
                                   "p_review": round(float(per_model[n][0][i]), 4)}
                               for n in model_names},
            "dissenting_models": [n for n in model_names if votes[n] == 0],
            "review_phrases":         _build_phrases(texts[i], reasons_by_model, 1),
            "nonreview_phrases":      _build_phrases(texts[i], reasons_by_model, 0),
            "dl_review_sentences":    dl_rev,
            "dl_nonreview_sentences": dl_nonrev,
        })
    return results

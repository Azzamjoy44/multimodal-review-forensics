import os
import re
import difflib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from load_reviews import get_reviews_by_product

# ---------------------------------------------------------------------------
# Model registry — lazy-loaded on first use
# ---------------------------------------------------------------------------
_model_cache = {}

def _load_model(key, path):
    if key in _model_cache:
        return _model_cache[key]
    model = None
    if os.path.exists(path):
        try:
            import joblib
            model = joblib.load(path)
        except Exception as e:
            print(f"Warning: could not load {path} — {e}")
    _model_cache[key] = model
    return model

FAKE_MODEL_NAMES      = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
SENTIMENT_MODEL_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
FAKE_CLASS_NAMES      = ("genuine", "fake")      # (negative, positive)
SENTIMENT_CLASS_NAMES = ("negative", "positive") # (negative, positive)

# ---------------------------------------------------------------------------
# Lightweight pure-Python tokenizer — reads Keras tokenizer JSON without TF
# ---------------------------------------------------------------------------
class _KerasTokenizer:
    """Minimal reimplementation of keras.preprocessing.text.Tokenizer for inference."""
    def __init__(self, word_index: dict, filters: str, lower: bool, num_words: int | None):
        self.word_index = word_index
        self._filters   = str.maketrans("", "", filters)
        self.lower      = lower
        self.num_words  = num_words  # embedding vocab size — indices >= this are out of bounds

    @classmethod
    def from_json(cls, path: str):
        import json
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)["config"]
        word_index = cfg["word_index"]
        if isinstance(word_index, str):
            word_index = json.loads(word_index)
        return cls(
            word_index=word_index,
            filters=cfg.get("filters", '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'),
            lower=cfg.get("lower", True),
            num_words=cfg.get("num_words"),
        )

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


def _lstm_pad(seqs, maxlen):
    """Post-pad / post-truncate sequences to fixed length (float32 for ONNX)."""
    out = np.zeros((len(seqs), maxlen), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = seq[:maxlen]
        out[i, :len(seq)] = seq
    return out


# ---------------------------------------------------------------------------
# LSTM ONNX models — thread-safe, no TensorFlow required at inference time
# ---------------------------------------------------------------------------
LSTM_MAX_LEN = FAKE_LSTM_MAX_LEN = 100
_DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")

_lstm_sess      = None
_lstm_tok       = None
_lstm_loaded    = False

_fake_lstm_sess   = None
_fake_lstm_tok    = None
_fake_lstm_loaded = False


def _load_lstm():
    global _lstm_sess, _lstm_tok, _lstm_loaded
    if _lstm_loaded:
        return _lstm_sess, _lstm_tok
    onnx_path = os.path.join(_DATA_DIR, "sentiment_lstm.onnx")
    tok_path  = os.path.join(_DATA_DIR, "sentiment_lstm_tokenizer.json")
    if os.path.exists(onnx_path) and os.path.exists(tok_path):
        try:
            import onnxruntime as ort
            _lstm_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            _lstm_tok  = _KerasTokenizer.from_json(tok_path)
            print("LSTM sentiment model loaded (ONNX).")
        except Exception as e:
            print(f"Warning: could not load LSTM sentiment ONNX model — {e}")
    _lstm_loaded = True
    return _lstm_sess, _lstm_tok


def _load_fake_lstm():
    global _fake_lstm_sess, _fake_lstm_tok, _fake_lstm_loaded
    if _fake_lstm_loaded:
        return _fake_lstm_sess, _fake_lstm_tok
    onnx_path = os.path.join(_DATA_DIR, "fake_lstm.onnx")
    tok_path  = os.path.join(_DATA_DIR, "fake_lstm_tokenizer.json")
    if os.path.exists(onnx_path) and os.path.exists(tok_path):
        try:
            import onnxruntime as ort
            _fake_lstm_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            _fake_lstm_tok  = _KerasTokenizer.from_json(tok_path)
            print("Fake-detection LSTM model loaded (ONNX).")
        except Exception as e:
            print(f"Warning: could not load fake LSTM ONNX model — {e}")
    _fake_lstm_loaded = True
    return _fake_lstm_sess, _fake_lstm_tok


def _run_fake_lstm(texts):
    sess, tokenizer = _load_fake_lstm()
    if sess is None or tokenizer is None:
        return [{"label": None, "confidence": None,
                 "reasons": ["Fake LSTM model not available"]} for _ in texts]
    try:
        padded = _lstm_pad(tokenizer.texts_to_sequences(texts), FAKE_LSTM_MAX_LEN)
        probs  = sess.run(None, {"input_layer": padded})[0].flatten()

        results = []
        vocab   = tokenizer.word_index
        for i, prob in enumerate(probs):
            label      = 1 if prob > 0.5 else 0
            confidence = round(float(prob if label == 1 else 1 - prob) * 100)
            label_str  = FAKE_CLASS_NAMES[label]
            known = [w for w in texts[i].lower().split()
                     if w in vocab and vocab[w] < 20_000
                     and w not in ENGLISH_STOP_WORDS][:3]
            reasons = ([f'"{w}" is associated with {label_str} reviews' for w in known]
                       or [f"Sequential word patterns indicate {label_str} review"])
            results.append({"label": label, "confidence": confidence, "reasons": reasons})
        return results
    except Exception as e:
        print(f"Warning: Fake LSTM prediction failed — {e}")
        return [{"label": None, "confidence": None, "reasons": ["Prediction error"]}
                for _ in texts]


FAKE_DISTILBERT_ONNX_DIR  = os.path.join(os.path.dirname(__file__), "data", "fake_distilbert_onnx")
_fake_distilbert_pipeline = None
_fake_distilbert_loaded   = False

def _load_fake_distilbert_onnx():
    global _fake_distilbert_pipeline, _fake_distilbert_loaded
    if _fake_distilbert_loaded:
        return _fake_distilbert_pipeline
    if os.path.isdir(FAKE_DISTILBERT_ONNX_DIR):
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import pipeline, AutoTokenizer
            tokenizer              = AutoTokenizer.from_pretrained(FAKE_DISTILBERT_ONNX_DIR)
            ort_model              = ORTModelForSequenceClassification.from_pretrained(
                FAKE_DISTILBERT_ONNX_DIR, provider="CPUExecutionProvider"
            )
            _fake_distilbert_pipeline = pipeline(
                "text-classification", model=ort_model, tokenizer=tokenizer, device=-1
            )
            print("Fake-detection DistilBERT ONNX model loaded.")
        except Exception as e:
            print(f"Warning: could not load fake DistilBERT ONNX model — {e}")
    _fake_distilbert_loaded = True
    return _fake_distilbert_pipeline


def _run_fake_distilbert_onnx(texts):
    """Run the DistilBERT ONNX fake-detection model on a list of texts."""
    pipe = _load_fake_distilbert_onnx()
    if pipe is None:
        return [{"label": None, "confidence": None,
                 "reasons": ["Fake DistilBERT ONNX model not available"]} for _ in texts]
    try:
        outputs = pipe(texts, truncation=True, max_length=128, batch_size=32)
        results = []
        for i, out in enumerate(outputs):
            label      = 1 if out["label"] == "LABEL_1" else 0
            confidence = round(float(out["score"]) * 100)
            label_str  = FAKE_CLASS_NAMES[1] if label == 1 else FAKE_CLASS_NAMES[0]
            words = [w for w in texts[i].lower().split()
                     if len(w) > 3 and w.isalpha()][:3]
            reasons = ([f'"{w}" is associated with {label_str} reviews' for w in words]
                       or [f"Contextual patterns indicate {label_str} review"])
            results.append({"label": label, "confidence": confidence, "reasons": reasons})
        return results
    except Exception as e:
        print(f"Warning: Fake DistilBERT ONNX prediction failed — {e}")
        return [{"label": None, "confidence": None, "reasons": ["Prediction error"]}
                for _ in texts]


BERT_MODEL_DIR  = os.path.join(os.path.dirname(__file__), "data", "bert_sentiment_model")
BERT_MAX_LEN    = 128
_bert_model     = None
_bert_tokenizer = None
_bert_loaded    = False

def _load_bert():
    global _bert_model, _bert_tokenizer, _bert_loaded
    if _bert_loaded:
        return _bert_model, _bert_tokenizer
    if os.path.isdir(BERT_MODEL_DIR):
        try:
            from transformers import BertTokenizerFast, BertForSequenceClassification
            import torch
            _bert_tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_DIR)
            _bert_model     = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
            _bert_model.eval()
            print("BERT sentiment model loaded.")
        except Exception as e:
            print(f"Warning: could not load BERT model — {e}")
    _bert_loaded = True
    return _bert_model, _bert_tokenizer


DISTILBERT_ONNX_DIR  = os.path.join(os.path.dirname(__file__), "data", "distilbert_sentiment_onnx")
DISTILBERT_MAX_LEN    = 128
_distilbert_sess      = None
_distilbert_tokenizer = None
_distilbert_loaded    = False

def _load_distilbert_onnx():
    """Load the sentiment DistilBERT directly via onnxruntime + tokenizers.

    NOTE: this deliberately does NOT use optimum/transformers `pipeline` — optimum imports
    TensorFlow for framework detection, which fails on a protobuf gencode/runtime mismatch
    and silently disabled this model. Raw onnxruntime (like model_b / review_detector) has
    no TF dependency. Returns (session, tokenizer) or None.
    """
    global _distilbert_sess, _distilbert_tokenizer, _distilbert_loaded
    if _distilbert_loaded:
        return (_distilbert_sess, _distilbert_tokenizer) if _distilbert_sess is not None else None
    onnx = os.path.join(DISTILBERT_ONNX_DIR, "model.onnx")
    if os.path.isfile(onnx):
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            _distilbert_sess      = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
            _distilbert_tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_ONNX_DIR)
            print("DistilBERT ONNX sentiment model loaded.")
        except Exception as e:
            print(f"Warning: could not load DistilBERT ONNX model — {e}")
            _distilbert_sess = None
    _distilbert_loaded = True
    return (_distilbert_sess, _distilbert_tokenizer) if _distilbert_sess is not None else None


def _run_distilbert_onnx_sentiment(texts):
    """Run the sentiment DistilBERT ONNX model on a list of texts (raw onnxruntime)."""
    loaded = _load_distilbert_onnx()
    if loaded is None:
        return [{"label": None, "confidence": None,
                 "reasons": ["DistilBERT ONNX model not available"]} for _ in texts]
    sess, tokenizer = loaded
    try:
        import numpy as np
        results, B = [], 32
        for s in range(0, len(texts), B):
            batch = texts[s:s + B]
            enc = tokenizer(batch, truncation=True, padding=True,
                            max_length=DISTILBERT_MAX_LEN, return_tensors="np")
            feed = {"input_ids": enc["input_ids"].astype(np.int64),
                    "attention_mask": enc["attention_mask"].astype(np.int64)}
            logits = np.asarray(sess.run(None, feed)[0], dtype=np.float64)   # [b, 2]
            ex     = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs  = ex / ex.sum(axis=1, keepdims=True)                       # softmax over 2 classes
            for j, t in enumerate(batch):
                label      = int(probs[j].argmax())     # 0 = negative, 1 = positive
                confidence = round(float(probs[j, label]) * 100)
                label_str  = "positive" if label == 1 else "negative"
                words = [w for w in t.lower().split() if len(w) > 3 and w.isalpha()][:3]
                reasons = ([f'"{w}" is associated with {label_str} sentiment' for w in words]
                           or [f"Contextual patterns indicate {label_str} sentiment"])
                results.append({"label": label, "confidence": confidence, "reasons": reasons})
        return results
    except Exception as e:
        print(f"Warning: DistilBERT ONNX prediction failed — {e}")
        return [{"label": None, "confidence": None, "reasons": ["DistilBERT prediction error"]}
                for _ in texts]


def _run_bert_sentiment(texts):
    """Run the BERT model on a list of texts. Returns one result dict per text."""
    import torch
    model, tokenizer = _load_bert()
    if model is None or tokenizer is None:
        return [{"label": None, "confidence": None,
                 "reasons": ["BERT model not available"]} for _ in texts]
    try:
        results = []
        batch_size = 32
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=BERT_MAX_LEN,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=1).numpy()
            labels = probs.argmax(axis=1)

            for i in range(len(batch_texts)):
                label      = int(labels[i])
                confidence = round(float(probs[i][label]) * 100)
                label_str  = "positive" if label == 1 else "negative"
                words = [w for w in batch_texts[i].lower().split()
                         if w in tokenizer.vocab and len(w) > 3][:3]
                reasons = ([f'"{w}" is associated with {label_str} sentiment' for w in words]
                           or [f"Contextual word patterns indicate {label_str} sentiment"])
                results.append({"label": label, "confidence": confidence, "reasons": reasons})
        return results
    except Exception as e:
        print(f"Warning: BERT prediction failed — {e}")
        return [{"label": None, "confidence": None, "reasons": ["BERT prediction error"]}
                for _ in texts]


def _run_lstm_sentiment(texts):
    sess, tokenizer = _load_lstm()
    if sess is None or tokenizer is None:
        return [{"label": None, "confidence": None,
                 "reasons": ["LSTM model not available"]} for _ in texts]
    try:
        padded = _lstm_pad(tokenizer.texts_to_sequences(texts), LSTM_MAX_LEN)
        probs  = sess.run(None, {"input_layer": padded})[0].flatten()

        results = []
        vocab   = tokenizer.word_index
        for i, prob in enumerate(probs):
            label      = 1 if prob > 0.5 else 0
            confidence = round(float(prob if label == 1 else 1 - prob) * 100)
            label_str  = "positive" if label == 1 else "negative"
            known = [w for w in texts[i].lower().split()
                     if w in vocab and vocab[w] < 20_000
                     and w not in ENGLISH_STOP_WORDS][:3]
            reasons = ([f'"{w}" is associated with {label_str} sentiment' for w in known]
                       or [f"Sequential word patterns indicate {label_str} sentiment"])
            results.append({"label": label, "confidence": confidence, "reasons": reasons})
        return results
    except Exception as e:
        print(f"Warning: LSTM sentiment prediction failed — {e}")
        return [{"label": None, "confidence": None, "reasons": ["LSTM prediction error"]}
                for _ in texts]


def _extract_reasons(pipeline, vec_sparse, feature_names, label, class_names, n=3):
    """Return top-n word reasons for the prediction using feature contributions."""
    clf      = pipeline.named_steps["clf"]
    clf_type = type(clf).__name__
    vec      = vec_sparse.toarray()[0]
    neg_name, pos_name = class_names
    label_str = pos_name if label == 1 else neg_name

    if clf_type == "KNeighborsClassifier":
        return ["Classified by similarity to nearest training examples"]

    if clf_type == "LogisticRegression":
        importance = clf.coef_[0]
    elif clf_type in ("MultinomialNB", "BernoulliNB", "ComplementNB"):
        importance = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif clf_type == "MLPClassifier":
        W1 = clf.coefs_[0]                           # (n_features, n_hidden)
        importance = np.sqrt((W1 ** 2).sum(axis=1))  # L2 norm per feature
        if label == 0:
            importance = -importance
    elif hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_.copy()
        if label == 0:
            importance = -importance
    else:
        return [f"Overall pattern matches {label_str} examples"]

    contribs = vec * importance

    if label == 1:
        top_idx = contribs.argsort()[-n:][::-1]
        words   = [feature_names[j] for j in top_idx if contribs[j] > 0]
    else:
        top_idx = contribs.argsort()[:n]
        words   = [feature_names[j] for j in top_idx if contribs[j] < 0]
        # Fallback for sparse models (e.g. AdaBoost): surface any vocabulary words
        # from the review, ranked by feature importance (may all be 0 for AdaBoost)
        if not words:
            in_vocab = [j for j in range(len(vec)) if vec[j] > 0]
            if in_vocab:
                if hasattr(clf, "feature_importances_"):
                    in_vocab.sort(key=lambda j: -clf.feature_importances_[j])
                words = [feature_names[j] for j in in_vocab[:n]]

    if not words:
        return [f"Overall pattern matches {label_str} examples"]
    return [f'"{w}" is associated with {label_str} reviews' for w in words]


def _run_models(model_names, prefix, texts, class_names):
    """Run all models for a given prefix on a list of texts.
    Returns list of dicts (one per text): {model_name: {label, confidence, reasons}}
    """
    results = [{} for _ in texts]

    for name in model_names:
        path     = os.path.join(os.path.dirname(__file__), "data", f"{prefix}_{name}.joblib")
        pipeline = _load_model(f"{prefix}_{name}", path)

        if pipeline is None:
            for r in results:
                r[name] = {"label": None, "confidence": None,
                           "reasons": ["Model not trained yet — run train script first"]}
            continue

        try:
            labels        = pipeline.predict(texts)
            probs         = pipeline.predict_proba(texts)
            feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
            vecs          = pipeline.named_steps["tfidf"].transform(texts)

            for i in range(len(texts)):
                label      = int(labels[i])
                confidence = round(float(probs[i][label]) * 100)
                reasons    = _extract_reasons(
                    pipeline, vecs[i], feature_names, label, class_names)
                results[i][name] = {
                    "label":      label,
                    "confidence": confidence,
                    "reasons":    reasons,
                }
        except Exception as e:
            print(f"Warning: {prefix}_{name} failed — {e}")
            for r in results:
                r[name] = {"label": None, "confidence": None, "reasons": ["Prediction error"]}

    return results


# ---------------------------------------------------------------------------
# Promotional / hype phrases that are common in fake reviews.
#
# CHANGE: Removed generic positive adjectives ("amazing", "best", "perfect")
# because they appear constantly in genuine reviews and caused many false
# positives.  Only explicit call-to-action and advertorial phrases remain,
# plus extreme hype words that are rarely used in organic reviews.
# ---------------------------------------------------------------------------
PROMO_WORDS = [
    # Explicit call-to-action phrases — strong fake-review signal
    "must buy", "buy now", "give it a try", "highly recommend",
    "everyone should", "everyone should play", "don't hesitate",
    "you won't regret", "worth every penny", "best purchase",
    "life changing", "five stars",
    # Extreme hype words rarely used in organic writing
    "incredible", "outstanding", "phenomenal", "unbelievable",
]

# Simple word lists for text-rating mismatch detection (no ML needed)
POSITIVE_WORDS = ["great", "amazing", "love", "excellent", "fantastic",
                  "awesome", "perfect", "best", "wonderful", "brilliant"]
NEGATIVE_WORDS = ["terrible", "awful", "worst", "hate", "horrible",
                  "broken", "garbage", "useless", "bad", "disappointing",
                  "crashed", "clunky", "boring", "waste"]

# Similarity threshold for near-duplicate detection (0.0 = nothing alike, 1.0 = identical)
NEAR_DUPLICATE_THRESHOLD = 0.85

# ---------------------------------------------------------------------------
# Words that carry no specific information about the product being reviewed.
# Any word that could appear in a review of *any* game is generic.
# Game-specific words (character names, mechanics, place names) are not here.
# Used by Rule 9 to detect short reviews made entirely of filler/sentiment.
# ---------------------------------------------------------------------------
GENERIC_WORDS = {
    # Function words
    "a", "an", "the", "this", "that", "it", "is", "was", "are", "were",
    "i", "me", "my", "we", "you", "he", "she", "they", "and", "or",
    "but", "not", "no", "yes", "so", "very", "really", "in", "on",
    "at", "to", "of", "for", "with", "from", "its", "has", "have",
    "get", "got", "can", "will", "be", "been", "all", "out", "up",
    "if", "do", "did", "than", "then", "too", "also", "just", "now",
    # Generic sentiment / quality words (applicable to any product)
    "good", "great", "bad", "nice", "ok", "okay", "cool", "fun",
    "love", "like", "hate", "best", "worst", "ever", "never",
    "amazing", "awesome", "fantastic", "perfect", "incredible",
    "beautiful", "excellent", "terrible", "awful", "loved", "hated",
    # Generic review vocabulary
    "game", "games", "play", "played", "playing", "buy", "bought",
    "recommend", "worth", "price", "time", "hours", "solid", "decent",
}


# ---------------------------------------------------------------------------
# Helper: normalise text for duplicate comparison
# Strips punctuation, lowercases, and collapses whitespace so that minor
# formatting differences don't hide a true duplicate.
# ---------------------------------------------------------------------------
def _normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse spaces
    return text


# ---------------------------------------------------------------------------
# Pre-pass: compare every review against every other review to find
# exact duplicates and near-duplicates before individual scoring starts.
#
# Returns two dicts keyed by review_id:
#   exact_dup_of[id]   → list of review_ids that are exact duplicates
#   near_dup_of[id]    → list of review_ids that are near-duplicates
# ---------------------------------------------------------------------------
def _find_duplicates(reviews, context=None):
    """Find exact/near duplicates AMONG `reviews`, and additionally flag any review in
    `reviews` that duplicates one in `context` (already-scored/cached reviews from earlier
    incremental loads). Output dicts are keyed only by `reviews` ids (the scored set);
    their values may reference context ids. context-vs-context pairs are NOT recomputed."""
    exact_dup_of = {r["review_id"]: [] for r in reviews}
    near_dup_of  = {r["review_id"]: [] for r in reviews}

    normed = [(r["review_id"], _normalize(r.get("review_text", ""))) for r in reviews]

    # within the new batch: symmetric pairwise (unchanged behaviour when context is empty)
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            id_a, text_a = normed[i]
            id_b, text_b = normed[j]
            if text_a == text_b:
                exact_dup_of[id_a].append(id_b)
                exact_dup_of[id_b].append(id_a)
            elif difflib.SequenceMatcher(None, text_a, text_b).ratio() >= NEAR_DUPLICATE_THRESHOLD:
                near_dup_of[id_a].append(id_b)
                near_dup_of[id_b].append(id_a)

    # against cached context: one-directional (only flag the NEW reviews) -> O(new x context)
    if context:
        ctx = [(r["review_id"], _normalize(r.get("review_text", ""))) for r in context]
        for id_a, text_a in normed:
            for id_b, text_b in ctx:
                if text_a == text_b:
                    exact_dup_of[id_a].append(id_b)
                elif difflib.SequenceMatcher(None, text_a, text_b).ratio() >= NEAR_DUPLICATE_THRESHOLD:
                    near_dup_of[id_a].append(id_b)

    return exact_dup_of, near_dup_of


# ---------------------------------------------------------------------------
# Score a single review using the rule-based checks.
# exact_dups and near_dups are lists of review_ids found in the pre-pass.
# ---------------------------------------------------------------------------
def score_review(review, exact_dups=None, near_dups=None):
    """
    Apply rule-based checks to one review dict.

    exact_dups  — list of review_ids that are exact duplicates of this one
    near_dups   — list of review_ids that are near-duplicates of this one

    Returns:
      { "suspicion_score": int (0–100), "reasons": [str, ...] }
    """
    if exact_dups is None:
        exact_dups = []
    if near_dups is None:
        near_dups = []

    text = review.get("review_text", "")
    score = 0
    reasons = []

    # ------------------------------------------------------------------
    # Rule 1 — Too short
    #
    # CHANGE: Weights reduced (30 → 20 and 15 → 10) because short reviews
    # alone were the single biggest source of false positives.  Shortness
    # is still a signal, but it now needs other signals to cross the
    # MEDIUM threshold.  Rule 9 below adds extra weight when the content
    # is also entirely generic, recovering the true short-fake cases.
    # ------------------------------------------------------------------
    if len(text) < 20:
        score += 20
        reasons.append("Review is very short (under 20 characters)")
    elif len(text) < 50:
        score += 10
        reasons.append("Review is quite short (under 50 characters)")

    # ------------------------------------------------------------------
    # Rule 2 — Repeated words
    # ------------------------------------------------------------------
    words = re.findall(r"[a-z]+", text.lower())
    if words:
        unique_words = set(words)
        repeated = [w for w in unique_words if words.count(w) > 1]
        repeat_ratio = len(repeated) / len(unique_words)
        if repeat_ratio > 0.5:
            score += 30
            reasons.append("More than half of the unique words are repeated")
        elif repeat_ratio > 0.3:
            score += 15
            reasons.append("Several words are repeated across the review")

    # ------------------------------------------------------------------
    # Rule 3 — Promotional / hype language
    # ------------------------------------------------------------------
    text_lower = text.lower()
    found_promo = [p for p in PROMO_WORDS if p in text_lower]
    if len(found_promo) >= 3:
        score += 30
        reasons.append(f"Heavy promotional language: {found_promo}")
    elif len(found_promo) >= 1:
        score += 15
        reasons.append(f"Promotional wording detected: {found_promo}")

    # ------------------------------------------------------------------
    # Rule 4 — Excessive punctuation
    # ------------------------------------------------------------------
    exclamation_clusters = re.findall(r"[!?]{2,}", text)
    if len(exclamation_clusters) >= 3:
        score += 20
        reasons.append("Excessive punctuation clusters (e.g. !!!, ???)")
    elif len(exclamation_clusters) >= 1:
        score += 10
        reasons.append("Repeated punctuation detected")

    # ------------------------------------------------------------------
    # Rule 5 — Exact duplicate
    # Strong signal: two different reviewers writing word-for-word the
    # same text is very unlikely to happen organically.
    # ------------------------------------------------------------------
    if exact_dups:
        score += 40
        reasons.append(
            f"Exact duplicate review text found in: {exact_dups}"
        )

    # ------------------------------------------------------------------
    # Rule 6 — Near-duplicate
    # High similarity to another review suggests copy-paste behaviour.
    # Only fired when the review is NOT already flagged as an exact dup.
    # ------------------------------------------------------------------
    if near_dups and not exact_dups:
        score += 25
        reasons.append(
            f"Review is very similar to another review: {near_dups}"
        )

    # ------------------------------------------------------------------
    # Rule 7 — Text-rating mismatch
    # Count positive and negative words, then check whether the tone
    # matches the star rating.  Ratings 1–2 are "low", 4–5 are "high".
    # ------------------------------------------------------------------
    pos_count = sum(1 for w in POSITIVE_WORDS if w in text_lower)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
    rating = review.get("rating", 3)
    if rating is None:                       # reviews without a star rating (e.g. labeled-Amazon set)
        rating = 3                           # treat as neutral so the rating-mismatch rules don't fire/crash

    if neg_count >= 2 and rating >= 4:
        score += 20
        reasons.append(
            f"Text sounds negative ({neg_count} negative words) but rating is {rating}/5"
        )
    elif pos_count >= 2 and rating <= 2:
        score += 20
        reasons.append(
            f"Text sounds positive ({pos_count} positive words) but rating is {rating}/5"
        )

    # ------------------------------------------------------------------
    # Rule 8 — Gibberish / symbol-only text
    #
    # NEW: Reviews made almost entirely of emoji, punctuation, or other
    # non-alphabetic characters contain no meaningful information.
    # We check the ratio of actual letters to total characters.
    # Threshold < 0.4 means fewer than 40 % of characters are letters.
    # Only checked when the text is long enough to rule out Rule 1 first
    # (very short texts are already penalised there).
    # ------------------------------------------------------------------
    if len(text) > 3:
        alpha_ratio = len(re.findall(r"[a-zA-Z]", text)) / len(text)
        if alpha_ratio < 0.4:
            score += 25
            reasons.append(
                f"Text is mostly symbols or non-alphabetic characters "
                f"({alpha_ratio:.0%} letters)"
            )

    # ------------------------------------------------------------------
    # Rule 9 — Short review with only generic words (combined signal)
    #
    # NEW: A short review whose every word could appear in a review of
    # *any* game (e.g. "good game", "amazing!", "best ever") provides
    # no real information and is a common fake-review pattern.
    # This fires only when the review is under 50 characters AND contains
    # zero words that are specific to the product being reviewed.
    # This compensates for the reduced Rule 1 weight by targeting the
    # genuine suspicious short-review cases more precisely.
    # ------------------------------------------------------------------
    if len(text) < 50 and words:
        non_generic = [w for w in words if w not in GENERIC_WORDS]
        if len(non_generic) == 0:
            score += 20
            reasons.append(
                "Short review contains only generic words — "
                "no product-specific information"
            )

    # Cap at 100
    score = min(score, 100)

    if not reasons:
        reasons.append("No obvious suspicious signals detected")

    return {
        "suspicion_score": score,
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Core scoring function — works on ANY list of review dicts.
# Called by both sample mode and live Steam mode.
# ---------------------------------------------------------------------------
def _split_sentences(text):
    """Split review text into sentences using punctuation boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if len(s.strip()) > 10]


def _get_dl_suspicious_sentences(text, fake_lstm_result, fake_distilbert_result):
    """
    Run sentence-level analysis through LSTM and DistilBERT when they predict
    the overall review is fake (confidence >= 70%). Returns sentences that
    individually score above threshold so the frontend can highlight them.
    """
    lstm_fake   = fake_lstm_result.get("label") == 1
    distil_fake = fake_distilbert_result.get("label") == 1

    if not lstm_fake and not distil_fake:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    # sentence → set of model names that flagged it
    sentence_models: dict = {}

    if lstm_fake:
        sess, tokenizer = _load_fake_lstm()
        if sess and tokenizer:
            try:
                padded = _lstm_pad(tokenizer.texts_to_sequences(sentences), FAKE_LSTM_MAX_LEN)
                probs  = sess.run(None, {"input_layer": padded})[0].flatten()
                flagged = [i for i, p in enumerate(probs) if p > 0.5]
                if not flagged:
                    flagged = [int(probs.argmax())]
                for i in flagged:
                    sentence_models.setdefault(sentences[i], set()).add("LSTM")
            except Exception:
                pass

    if distil_fake:
        pipe = _load_fake_distilbert_onnx()
        if pipe:
            try:
                results = pipe(sentences, truncation=True, max_length=128, batch_size=32)
                flagged = [i for i, r in enumerate(results)
                           if r["label"] == "LABEL_1" and r["score"] > 0.5]
                if not flagged:
                    # Always highlight the sentence DistilBERT found most suspicious
                    scores  = [r["score"] if r["label"] == "LABEL_1" else 1 - r["score"]
                               for r in results]
                    flagged = [scores.index(max(scores))]
                for i in flagged:
                    sentence_models.setdefault(sentences[i], set()).add("DistilBERT")
            except Exception:
                pass

    return [
        {"sentence": s, "models": sorted(list(m))}
        for s, m in sentence_models.items()
    ]


def _get_dl_genuine_sentences(text, fake_lstm_result, fake_distilbert_result):
    """
    Mirror of _get_dl_suspicious_sentences but for GENUINE predictions.
    Finds the sentences most confidently classified as genuine (lowest fake probability)
    so the frontend can highlight them when LSTM/DistilBERT is clicked.
    """
    lstm_genuine   = fake_lstm_result.get("label") == 0
    distil_genuine = fake_distilbert_result.get("label") == 0

    if not lstm_genuine and not distil_genuine:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    sentence_models: dict = {}

    if lstm_genuine:
        sess, tokenizer = _load_fake_lstm()
        if sess and tokenizer:
            try:
                padded = _lstm_pad(tokenizer.texts_to_sequences(sentences), FAKE_LSTM_MAX_LEN)
                probs  = sess.run(None, {"input_layer": padded})[0].flatten()
                # Low fake probability = confidently genuine
                flagged = [i for i, p in enumerate(probs) if p < 0.3]
                if not flagged:
                    flagged = [int(probs.argmin())]
                for i in flagged:
                    sentence_models.setdefault(sentences[i], set()).add("LSTM")
            except Exception:
                pass

    if distil_genuine:
        pipe = _load_fake_distilbert_onnx()
        if pipe:
            try:
                results = pipe(sentences, truncation=True, max_length=128, batch_size=32)
                flagged = [i for i, r in enumerate(results)
                           if r["label"] == "LABEL_0" and r["score"] > 0.7]
                if not flagged:
                    scores  = [r["score"] if r["label"] == "LABEL_0" else 1 - r["score"]
                               for r in results]
                    flagged = [scores.index(max(scores))]
                for i in flagged:
                    sentence_models.setdefault(sentences[i], set()).add("DistilBERT")
            except Exception:
                pass

    return [
        {"sentence": s, "models": sorted(list(m)), "type": "genuine"}
        for s, m in sentence_models.items()
    ]


def _get_suspicious_phrases(text, fake_preds):
    """
    Return a list of {phrase, type} dicts for words/phrases in the text that
    are either promotional (from PROMO_WORDS) or flagged by the ML model.
    Used by the frontend to highlight suspicious language in the review.
    """
    text_lower = text.lower()
    phrases = []
    seen    = set()

    # Promotional / emotive words from the rule-based system
    for phrase in PROMO_WORDS:
        if phrase in text_lower and phrase not in seen:
            phrases.append({"phrase": phrase, "type": "promotional", "models": ["Rule-based"]})
            seen.add(phrase)

    # Top words from all fake-detection ML models — collect all models per phrase
    phrase_models: dict = {}  # phrase_lower → set of model names
    model_display = {
        "knn": "KNN", "rf": "Random Forest", "dt": "Decision Tree",
        "lr": "LR", "nb": "Naïve Bayes", "xgb": "XGBoost",
        "adaboost": "AdaBoost", "mlp": "MLP",
        "lstm": "LSTM", "distilbert": "DistilBERT",
    }
    for model_name in FAKE_MODEL_NAMES:   # sklearn only — LSTM/DistilBERT flag sentences, not words
        model_result = fake_preds.get(model_name, {})
        if model_result.get("label") == 1:
            for reason in (model_result.get("reasons") or []):
                m = re.search(r'"([^"]+)"', reason)
                if m:
                    w = m.group(1).strip()
                    if w.lower() in text_lower and w.lower() not in ENGLISH_STOP_WORDS:
                        phrase_models.setdefault(w.lower(), {"phrase": w, "models": set()})
                        phrase_models[w.lower()]["models"].add(
                            model_display.get(model_name, model_name.upper())
                        )

    for w_lower, info in phrase_models.items():
        if w_lower not in seen:
            phrases.append({
                "phrase":  info["phrase"],
                "type":    "ml_flagged",
                "models":  sorted(list(info["models"])),
            })
            seen.add(w_lower)

    return phrases


def _get_genuine_phrases(text, fake_preds):
    """
    Same as _get_suspicious_phrases but for models that predicted GENUINE (label == 0).
    Returns [{phrase, type:'genuine_signal', models:[...]}] — one entry per unique word.
    Stored separately so the frontend can highlight them only when a genuine model is selected.
    """
    text_lower = text.lower()
    model_display = {
        "knn": "KNN", "rf": "Random Forest", "dt": "Decision Tree",
        "lr": "LR", "nb": "Naïve Bayes", "xgb": "XGBoost",
        "adaboost": "AdaBoost", "mlp": "MLP",
        "lstm": "LSTM", "distilbert": "DistilBERT",
    }
    phrase_models: dict = {}
    for model_name in FAKE_MODEL_NAMES:   # sklearn only — LSTM/DistilBERT flag sentences, not words
        model_result = fake_preds.get(model_name, {})
        if model_result.get("label") == 0:
            for reason in (model_result.get("reasons") or []):
                m = re.search(r'"([^"]+)"', reason)
                if m:
                    w = m.group(1).strip()
                    if w.lower() in text_lower and w.lower() not in ENGLISH_STOP_WORDS:
                        phrase_models.setdefault(w.lower(), {"phrase": w, "models": set()})
                        phrase_models[w.lower()]["models"].add(
                            model_display.get(model_name, model_name.upper())
                        )
    return [
        {"phrase": info["phrase"], "type": "genuine_signal", "models": sorted(list(info["models"]))}
        for info in phrase_models.values()
    ]


def _get_dl_sentiment_sentences(text, lstm_sent_result, distilbert_sent_result, target_label):
    """
    Sentence-level sentiment analysis for LSTM and DistilBERT.
    target_label: 1 = positive, 0 = negative
    Returns [{sentence, models, type}]
    """
    lstm_match   = lstm_sent_result.get("label") == target_label
    distil_match = distilbert_sent_result.get("label") == target_label

    if not lstm_match and not distil_match:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    sent_type      = "positive_sent" if target_label == 1 else "negative_sent"
    sentence_models: dict = {}

    if lstm_match:
        try:
            results = _run_lstm_sentiment(sentences)
            flagged = [i for i, r in enumerate(results) if r.get("label") == target_label]
            if not flagged:
                confs   = [r.get("confidence", 0) if r.get("label") == target_label else 0
                           for r in results]
                flagged = [int(np.argmax(confs))]
            for i in flagged:
                sentence_models.setdefault(sentences[i], set()).add("LSTM")
        except Exception:
            pass

    if distil_match:
        try:
            results = _run_distilbert_onnx_sentiment(sentences)
            flagged = [i for i, r in enumerate(results) if r.get("label") == target_label]
            if not flagged:
                confs   = [r.get("confidence", 0) if r.get("label") == target_label else 0
                           for r in results]
                flagged = [int(np.argmax(confs))]
            for i in flagged:
                sentence_models.setdefault(sentences[i], set()).add("DistilBERT")
        except Exception:
            pass

    return [
        {"sentence": s, "models": sorted(list(m)), "type": sent_type}
        for s, m in sentence_models.items()
    ]


def _get_sentiment_phrases(text, sent_preds, target_label):
    """
    Extract words from sentiment models that predicted target_label.
    target_label: 1 = positive, 0 = negative
    Returns [{phrase, type, models}]
    """
    text_lower = text.lower()
    model_display = {
        "knn": "KNN", "rf": "Random Forest", "dt": "Decision Tree",
        "lr": "LR", "nb": "Naïve Bayes", "xgb": "XGBoost",
        "adaboost": "AdaBoost", "mlp": "MLP",
        "lstm": "LSTM", "distilbert": "DistilBERT",
    }
    phrase_type = "positive_signal" if target_label == 1 else "negative_signal"
    phrase_models: dict = {}
    for model_name, model_result in sent_preds.items():
        if model_result.get("label") == target_label:
            for reason in (model_result.get("reasons") or []):
                m = re.search(r'"([^"]+)"', reason)
                if m:
                    w = m.group(1).strip()
                    if w.lower() in text_lower and w.lower() not in ENGLISH_STOP_WORDS:
                        phrase_models.setdefault(w.lower(), {"phrase": w, "models": set()})
                        phrase_models[w.lower()]["models"].add(
                            model_display.get(model_name, model_name.upper())
                        )
    return [
        {"phrase": info["phrase"], "type": phrase_type, "models": sorted(list(info["models"]))}
        for info in phrase_models.values()
    ]


# ---------------------------------------------------------------------------
# Yelp fake detection — separate model set trained on YelpZip/Chi/NYC
# ---------------------------------------------------------------------------
YELP_FAKE_MODEL_NAMES = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
YELP_MODEL_DIR        = os.path.join(_DATA_DIR, "yelp_fake_sklearn_models")
# Deep-learning model variants the frontend can choose between for Yelp fake detection.
YELP_DL_VARIANTS = ("baseline", "focal", "contrastive")
YELP_LSTM_DIRS = {
    "baseline":    os.path.join(_DATA_DIR, "yelp_fake_lstm_onnx"),
    "focal":       os.path.join(_DATA_DIR, "yelp_fake_lstm_focal_onnx"),
    "contrastive": os.path.join(_DATA_DIR, "yelp_fake_lstm_contrastive_onnx"),
}
YELP_DISTILBERT_DIRS = {
    "baseline":    os.path.join(_DATA_DIR, "yelp_fake_distilbert_onnx"),
    "focal":       os.path.join(_DATA_DIR, "yelp_fake_distilbert_focal_onnx"),
    "contrastive": os.path.join(_DATA_DIR, "yelp_fake_distilbert_contrastive_onnx"),
}
# Backward-compatible aliases (default = baseline)
YELP_LSTM_DIR         = YELP_LSTM_DIRS["baseline"]
YELP_DISTILBERT_DIR   = YELP_DISTILBERT_DIRS["baseline"]
YELP_LSTM_MAX_LEN     = 250  # must match training (MAX_LEN=250 in notebook)
YELP_BERT_MAX_LEN     = 256  # must match training (MAX_LEN=256 in DistilBERT notebooks)

_yelp_sklearn_thresholds        = None
_yelp_sklearn_thresholds_loaded = False

# Per-variant caches: variant -> loaded objects (loaded lazily on first use)
_yelp_lstm_cache       = {}   # variant -> (sess, tok, threshold)
_yelp_distilbert_cache = {}   # variant -> (pipeline, threshold)


def yelp_dl_variants_available():
    """Variants whose LSTM *and* DistilBERT folders both exist on disk.
    The frontend uses this to only offer selectable variants."""
    avail = []
    for v in YELP_DL_VARIANTS:
        if os.path.isdir(YELP_LSTM_DIRS[v]) and os.path.isdir(YELP_DISTILBERT_DIRS[v]):
            avail.append(v)
    return avail or ["baseline"]


def _load_yelp_sklearn_thresholds():
    global _yelp_sklearn_thresholds, _yelp_sklearn_thresholds_loaded
    if _yelp_sklearn_thresholds_loaded:
        return _yelp_sklearn_thresholds
    path = os.path.join(YELP_MODEL_DIR, "yelp_fake_thresholds.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            _yelp_sklearn_thresholds = json.load(f)
    _yelp_sklearn_thresholds_loaded = True
    return _yelp_sklearn_thresholds


def _load_yelp_lstm(variant: str = "baseline"):
    """Load the Yelp fake LSTM for the given variant (baseline/focal/contrastive).
    Cached per variant. Returns (session, tokenizer, threshold)."""
    if variant not in YELP_LSTM_DIRS:
        variant = "baseline"
    if variant in _yelp_lstm_cache:
        return _yelp_lstm_cache[variant]

    model_dir = YELP_LSTM_DIRS[variant]
    base = os.path.basename(model_dir).replace("_onnx", "")   # e.g. yelp_fake_lstm_focal
    onnx_path = os.path.join(model_dir, base + ".onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(model_dir, "model.onnx")
    tok_path = os.path.join(model_dir, base + "_tokenizer.json")
    if not os.path.exists(tok_path) and os.path.isdir(model_dir):
        for fn in os.listdir(model_dir):
            if fn.endswith("_tokenizer.json"):
                tok_path = os.path.join(model_dir, fn)
                break
    thresh_path = os.path.join(model_dir, base + "_threshold.json")
    if not os.path.exists(thresh_path):
        thresh_path = os.path.join(model_dir, "threshold.json")

    sess = tok = None
    threshold = 0.57
    if os.path.exists(onnx_path) and os.path.exists(tok_path):
        try:
            import onnxruntime as ort, json
            sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            tok  = _KerasTokenizer.from_json(tok_path)
            if os.path.exists(thresh_path):
                with open(thresh_path) as f:
                    threshold = json.load(f).get("threshold", 0.57)
            print(f"Yelp fake LSTM [{variant}] loaded (ONNX).")
        except Exception as e:
            print(f"Warning: could not load Yelp LSTM [{variant}] — {e}")

    _yelp_lstm_cache[variant] = (sess, tok, threshold)
    return _yelp_lstm_cache[variant]


def _load_yelp_distilbert(variant: str = "baseline"):
    """Load the Yelp fake DistilBERT for the given variant. Cached per variant.
    Returns (pipeline, threshold)."""
    if variant not in YELP_DISTILBERT_DIRS:
        variant = "baseline"
    if variant in _yelp_distilbert_cache:
        return _yelp_distilbert_cache[variant]

    model_dir   = YELP_DISTILBERT_DIRS[variant]
    thresh_path = os.path.join(model_dir, "threshold.json")
    pipe = None
    threshold = 0.51
    if os.path.isdir(model_dir):
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification
            from transformers import pipeline as hf_pipeline, AutoTokenizer
            import json
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            ort_model = ORTModelForSequenceClassification.from_pretrained(
                model_dir, provider="CPUExecutionProvider"
            )
            pipe = hf_pipeline(
                "text-classification", model=ort_model, tokenizer=tokenizer, device=-1
            )
            if os.path.exists(thresh_path):
                with open(thresh_path) as f:
                    threshold = json.load(f).get("threshold", 0.51)
            print(f"Yelp fake DistilBERT [{variant}] loaded (ONNX).")
        except Exception as e:
            print(f"Warning: could not load Yelp DistilBERT [{variant}] — {e}")

    _yelp_distilbert_cache[variant] = (pipe, threshold)
    return _yelp_distilbert_cache[variant]


def _run_yelp_sklearn_models(texts):
    """Run the 8 Yelp sklearn models using their val-tuned thresholds."""
    thresholds = _load_yelp_sklearn_thresholds() or {}
    results    = [{} for _ in texts]

    for name in YELP_FAKE_MODEL_NAMES:
        pipeline  = _load_model(
            f"yelp_fake_{name}",
            os.path.join(YELP_MODEL_DIR, f"yelp_fake_{name}.joblib"),
        )
        threshold = thresholds.get(name, 0.5)

        if pipeline is None:
            for r in results:
                r[name] = {"label": None, "confidence": None,
                           "reasons": ["Yelp model not available"]}
            continue

        try:
            probs         = pipeline.predict_proba(texts)
            fake_probs    = probs[:, 1]
            feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
            vecs          = pipeline.named_steps["tfidf"].transform(texts)

            for i in range(len(texts)):
                label      = 1 if fake_probs[i] >= threshold else 0
                confidence = round(float(fake_probs[i] if label == 1 else 1 - fake_probs[i]) * 100)
                reasons    = _extract_reasons(
                    pipeline, vecs[i], feature_names, label, FAKE_CLASS_NAMES)
                results[i][name] = {"label": label, "confidence": confidence, "reasons": reasons}
        except Exception as e:
            print(f"Warning: yelp_fake_{name} failed — {e}")
            for r in results:
                r[name] = {"label": None, "confidence": None, "reasons": ["Prediction error"]}

    return results


def _run_yelp_lstm(texts, variant: str = "baseline"):
    sess, tokenizer, threshold = _load_yelp_lstm(variant)
    if sess is None or tokenizer is None:
        return [{"label": None, "confidence": None,
                 "reasons": ["Yelp LSTM model not available"]} for _ in texts]
    try:
        padded = _lstm_pad(tokenizer.texts_to_sequences(texts), YELP_LSTM_MAX_LEN)
        probs  = sess.run(None, {"input_layer": padded})[0].flatten()
        results = []
        vocab   = tokenizer.word_index
        for i, prob in enumerate(probs):
            label      = 1 if prob >= threshold else 0
            confidence = round(float(prob if label == 1 else 1 - prob) * 100)
            label_str  = FAKE_CLASS_NAMES[label]
            known = [w for w in texts[i].lower().split()
                     if w in vocab and vocab[w] < 20_000
                     and w not in ENGLISH_STOP_WORDS][:3]
            reasons = ([f'"{w}" is associated with {label_str} reviews' for w in known]
                       or [f"Sequential word patterns indicate {label_str} review"])
            results.append({"label": label, "confidence": confidence, "reasons": reasons})
        return results
    except Exception as e:
        print(f"Warning: Yelp LSTM prediction failed — {e}")
        return [{"label": None, "confidence": None, "reasons": ["Prediction error"]}
                for _ in texts]


def _run_yelp_distilbert(texts, variant: str = "baseline"):
    pipe, threshold = _load_yelp_distilbert(variant)
    if pipe is None:
        return [{"label": None, "confidence": None,
                 "reasons": ["Yelp DistilBERT not available"]} for _ in texts]
    try:
        outputs  = pipe(texts, truncation=True, max_length=YELP_BERT_MAX_LEN, batch_size=32)
        results  = []
        for i, out in enumerate(outputs):
            raw_prob   = float(out["score"]) if out["label"] == "LABEL_1" else 1.0 - float(out["score"])
            label      = 1 if raw_prob >= threshold else 0
            confidence = round((raw_prob if label == 1 else 1 - raw_prob) * 100)
            label_str  = FAKE_CLASS_NAMES[label]
            words = [w for w in texts[i].lower().split() if len(w) > 3 and w.isalpha()][:3]
            reasons = ([f'"{w}" is associated with {label_str} reviews' for w in words]
                       or [f"Contextual patterns indicate {label_str} review"])
            results.append({"label": label, "confidence": confidence, "reasons": reasons})
        return results
    except Exception as e:
        print(f"Warning: Yelp DistilBERT prediction failed — {e}")
        return [{"label": None, "confidence": None, "reasons": ["Prediction error"]}
                for _ in texts]


def _get_yelp_dl_suspicious_sentences(text, lstm_result, distilbert_result, variant: str = "baseline"):
    lstm_fake   = lstm_result.get("label") == 1
    distil_fake = distilbert_result.get("label") == 1
    if not lstm_fake and not distil_fake:
        return []
    sentences = _split_sentences(text)
    if not sentences:
        return []

    sentence_models: dict = {}
    _, _, lstm_thresh = _load_yelp_lstm(variant)

    if lstm_fake:
        sess, tokenizer, _ = _load_yelp_lstm(variant)
        if sess and tokenizer:
            try:
                padded  = _lstm_pad(tokenizer.texts_to_sequences(sentences), YELP_LSTM_MAX_LEN)
                probs   = sess.run(None, {"input_layer": padded})[0].flatten()
                flagged = [i for i, p in enumerate(probs) if p >= lstm_thresh]
                if not flagged:
                    flagged = [int(probs.argmax())]
                for i in flagged:
                    sentence_models.setdefault(sentences[i], set()).add("LSTM")
            except Exception:
                pass

    if distil_fake:
        pipe, distil_thresh = _load_yelp_distilbert(variant)
        if pipe:
            try:
                outs       = pipe(sentences, truncation=True, max_length=YELP_BERT_MAX_LEN, batch_size=32)
                fake_probs = [float(o["score"]) if o["label"] == "LABEL_1"
                              else 1.0 - float(o["score"]) for o in outs]
                flagged    = [i for i, p in enumerate(fake_probs) if p >= distil_thresh]
                if not flagged:
                    flagged = [int(np.argmax(fake_probs))]
                for i in flagged:
                    sentence_models.setdefault(sentences[i], set()).add("DistilBERT")
            except Exception:
                pass

    return [{"sentence": s, "models": sorted(list(m))} for s, m in sentence_models.items()]


def _get_yelp_dl_genuine_sentences(text, lstm_result, distilbert_result, variant: str = "baseline"):
    lstm_genuine   = lstm_result.get("label") == 0
    distil_genuine = distilbert_result.get("label") == 0
    if not lstm_genuine and not distil_genuine:
        return []
    sentences = _split_sentences(text)
    if not sentences:
        return []

    sentence_models: dict = {}

    if lstm_genuine:
        sess, tokenizer, _ = _load_yelp_lstm(variant)
        if sess and tokenizer:
            try:
                padded  = _lstm_pad(tokenizer.texts_to_sequences(sentences), YELP_LSTM_MAX_LEN)
                probs   = sess.run(None, {"input_layer": padded})[0].flatten()
                flagged = [i for i, p in enumerate(probs) if p < 0.3]
                if not flagged:
                    flagged = [int(probs.argmin())]
                for i in flagged:
                    sentence_models.setdefault(sentences[i], set()).add("LSTM")
            except Exception:
                pass

    if distil_genuine:
        pipe, _ = _load_yelp_distilbert(variant)
        if pipe:
            try:
                outs       = pipe(sentences, truncation=True, max_length=YELP_BERT_MAX_LEN, batch_size=32)
                fake_probs = [float(o["score"]) if o["label"] == "LABEL_1"
                              else 1.0 - float(o["score"]) for o in outs]
                flagged    = [i for i, p in enumerate(fake_probs) if p < 0.3]
                if not flagged:
                    flagged = [int(np.argmin(fake_probs))]
                for i in flagged:
                    sentence_models.setdefault(sentences[i], set()).add("DistilBERT")
            except Exception:
                pass

    return [{"sentence": s, "models": sorted(list(m)), "type": "genuine"}
            for s, m in sentence_models.items()]


# Per-review caches so neither switching DL variant NOR changing the review count
# (e.g. 50 -> 60 reviews for the same business) recomputes work already done:
#   - sklearn is variant-independent  -> cached per review TEXT
#   - LSTM/DistilBERT + the highlight spans depend on the variant -> cached per (TEXT, variant)
# Rule/duplicate scoring is set-dependent (cheap) and is always recomputed.
_yelp_sklearn_cache = {}   # text_key -> sklearn pred dict (8 models)
_yelp_dl_cache      = {}   # (text_key, variant) -> bundle of model-derived fields
_YELP_CACHE_MAX     = 5000


def _yelp_text_key(text):
    import hashlib
    return hashlib.md5(str(text).encode("utf-8", "ignore")).hexdigest()


def _yelp_cache_put(cache, k, v):
    if k not in cache and len(cache) >= _YELP_CACHE_MAX:
        cache.pop(next(iter(cache)))   # FIFO eviction
    cache[k] = v


def score_yelp_review_list(reviews, variant: str = "baseline", fake_override=None, dup_context=None):
    """Score Yelp reviews.

    fake_override: optional list (one per review) of Model-B bundles
        {fake_models, suspicious_phrases, genuine_phrases,
         dl_suspicious_sentences, dl_genuine_sentences}. When provided, the multi-modal
        Model B supplies the fake verdicts + highlights and the text-only Yelp baseline
        models are NOT run/loaded (the live Yelp section always passes this). Rule-based
        suspicion + duplicate detection are unaffected.

    Otherwise (offline/eval) the text-only baseline runs: `variant` selects which
    deep-learning weights back 'lstm'/'distilbert' ('baseline'|'focal'|'contrastive');
    the 8 sklearn models have no variants. Predictions are cached per review.
    """
    if not reviews:
        return []
    if variant not in YELP_DL_VARIANTS:
        variant = "baseline"

    texts = [r.get("review_text", "") for r in reviews]

    # Duplicate detection is set-dependent. `dup_context` = already-cached reviews from earlier
    # incremental loads, so a new review duplicating a cached one is still flagged.
    exact_dup_of, near_dup_of = _find_duplicates(reviews, context=dup_context)

    if fake_override is None:
        keys = [_yelp_text_key(t) for t in texts]
        # sklearn (variant-independent): compute only for texts not already cached
        skl_missing = [i for i, k in enumerate(keys) if k not in _yelp_sklearn_cache]
        if skl_missing:
            skl_res = _run_yelp_sklearn_models([texts[i] for i in skl_missing])
            for j, i in enumerate(skl_missing):
                _yelp_cache_put(_yelp_sklearn_cache, keys[i], skl_res[j])

        # DL (variant-dependent): compute only for (text, variant) not already cached
        dl_missing = [i for i, k in enumerate(keys) if (k, variant) not in _yelp_dl_cache]
        if dl_missing:
            m_texts  = [texts[i] for i in dl_missing]
            lstm_res = _run_yelp_lstm(m_texts, variant)
            bert_res = _run_yelp_distilbert(m_texts, variant)
            for j, i in enumerate(dl_missing):
                fake_preds = dict(_yelp_sklearn_cache[keys[i]])
                fake_preds["lstm"]       = lstm_res[j]
                fake_preds["distilbert"] = bert_res[j]
                bundle = {
                    "fake_models":             fake_preds,
                    "suspicious_phrases":      _get_suspicious_phrases(texts[i], fake_preds),
                    "genuine_phrases":         _get_genuine_phrases(texts[i], fake_preds),
                    "dl_suspicious_sentences": _get_yelp_dl_suspicious_sentences(
                        texts[i], lstm_res[j], bert_res[j], variant),
                    "dl_genuine_sentences":    _get_yelp_dl_genuine_sentences(
                        texts[i], lstm_res[j], bert_res[j], variant),
                }
                _yelp_cache_put(_yelp_dl_cache, (keys[i], variant), bundle)

    results = []
    for i, review in enumerate(reviews):
        rid         = review["review_id"]
        rule_result = score_review(review,
                                   exact_dups=exact_dup_of[rid],
                                   near_dups=near_dup_of[rid])
        bundle = fake_override[i] if fake_override is not None else _yelp_dl_cache[(keys[i], variant)]
        results.append({
            "review_id":          rid,
            "review_text":        review["review_text"],
            "rating":             review.get("rating"),
            "user_id":            review.get("user_id", ""),
            "time":               review.get("time", 0),
            "business_id":        review.get("business_id", ""),
            "ground_truth_label": review.get("ground_truth_label"),
            "suspicion_score":    rule_result["suspicion_score"],
            "rule_reasons":       rule_result["reasons"],
            "fake_models":             bundle["fake_models"],
            "suspicious_phrases":      bundle["suspicious_phrases"],
            "genuine_phrases":         bundle["genuine_phrases"],
            "dl_suspicious_sentences": bundle["dl_suspicious_sentences"],
            "dl_genuine_sentences":    bundle["dl_genuine_sentences"],
        })

    return results


def score_review_list(reviews, task: str = "fake", fake_override=None):
    """
    Score every review with the relevant models for the requested task.

    task: "fake"      — runs fake detection models only
          "sentiment" — runs sentiment models only

    fake_override: optional list (one per review) of Model-A output dicts
        {fake_models, suspicious_phrases, genuine_phrases,
         dl_suspicious_sentences, dl_genuine_sentences}. When provided (task="fake"),
        the broadened all-domain detector supplies the fake verdicts + highlights and
        the Amazon-specific fake models are NOT run/loaded. Rule-based suspicion and
        sentiment are unaffected.

    Returns a list of result dicts per review.
    """
    if not reviews:
        return []

    exact_dup_of, near_dup_of = _find_duplicates(reviews)
    texts = [r.get("review_text", "") for r in reviews]

    if task == "fake":
        if fake_override is not None:
            fake_preds            = [fo["fake_models"] for fo in fake_override]   # already incl. lstm+distilbert
            fake_lstm_preds = fake_distilbert_preds = None
        else:
            fake_preds            = _run_models(FAKE_MODEL_NAMES, "fake", texts, FAKE_CLASS_NAMES)
            fake_lstm_preds       = _run_fake_lstm(texts)
            fake_distilbert_preds = _run_fake_distilbert_onnx(texts)
        sentiment_preds       = [{} for _ in texts]
        lstm_preds            = [{"label": None, "confidence": None, "reasons": []} for _ in texts]
        distilbert_preds      = [{"label": None, "confidence": None, "reasons": []} for _ in texts]
    else:  # sentiment
        sentiment_preds       = _run_models(SENTIMENT_MODEL_NAMES, "sentiment", texts, SENTIMENT_CLASS_NAMES)
        lstm_preds            = _run_lstm_sentiment(texts)
        distilbert_preds      = _run_distilbert_onnx_sentiment(texts)
        fake_preds            = [{} for _ in texts]
        fake_lstm_preds       = [{"label": None, "confidence": None, "reasons": []} for _ in texts]
        fake_distilbert_preds = [{"label": None, "confidence": None, "reasons": []} for _ in texts]

    results = []
    for i, review in enumerate(reviews):
        rid    = review["review_id"]
        result = score_review(
            review,
            exact_dups=exact_dup_of[rid],
            near_dups=near_dup_of[rid],
        )
        if task == "fake" and fake_override is not None:
            fo      = fake_override[i]
            fp      = fo["fake_models"]              # already includes lstm + distilbert
            susp_ph = fo["suspicious_phrases"]
            genu_ph = fo["genuine_phrases"]
            dl_susp = fo["dl_suspicious_sentences"]
            dl_genu = fo["dl_genuine_sentences"]
        else:
            fake_preds[i]["lstm"]       = fake_lstm_preds[i]
            fake_preds[i]["distilbert"] = fake_distilbert_preds[i]
            fp      = fake_preds[i]
            susp_ph = _get_suspicious_phrases(review["review_text"], fp)
            genu_ph = _get_genuine_phrases(review["review_text"], fp)
            dl_susp = _get_dl_suspicious_sentences(review["review_text"], fp.get("lstm", {}), fp.get("distilbert", {}))
            dl_genu = _get_dl_genuine_sentences(review["review_text"], fp.get("lstm", {}), fp.get("distilbert", {}))
        sent = sentiment_preds[i]
        sent["lstm"]       = lstm_preds[i]
        # sent["bert"]     = bert_preds[i]  # commented out
        sent["distilbert"] = distilbert_preds[i]
        results.append({
            "review_id":          rid,
            "review_text":        review["review_text"],
            "rating":             review["rating"],
            "user_id":            review.get("user_id", ""),
            "time":               review.get("time", 0),
            "suspicion_score":    result["suspicion_score"],
            "rule_reasons":       result["reasons"],
            "fake_models":        fp,
            "sentiment_models":   sent,
            "sentiment_positive_phrases": _get_sentiment_phrases(review["review_text"], sent, 1),
            "sentiment_negative_phrases": _get_sentiment_phrases(review["review_text"], sent, 0),
            "sentiment_dl_positive_sentences": _get_dl_sentiment_sentences(
                review["review_text"], sent.get("lstm", {}), sent.get("distilbert", {}), 1),
            "sentiment_dl_negative_sentences": _get_dl_sentiment_sentences(
                review["review_text"], sent.get("lstm", {}), sent.get("distilbert", {}), 0),
            "suspicious_phrases":      susp_ph,
            "genuine_phrases":         genu_ph,
            "dl_suspicious_sentences": dl_susp,
            "dl_genuine_sentences":    dl_genu,
        })

    return results


# ---------------------------------------------------------------------------
# Convenience wrapper for sample-data mode: load by product name then score.
# ---------------------------------------------------------------------------
def score_reviews_for_product(product_name):
    """
    Load reviews from the sample JSON file and score them.
    Kept for backward compatibility — main.py sample mode uses this.
    """
    reviews = get_reviews_by_product(product_name)

    if not reviews:
        print(f"No reviews found for '{product_name}'.")
        return []

    return score_review_list(reviews)


# ---------------------------------------------------------------------------
# Quick demo — run this file directly:
#   python score_reviews.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    product = "Cyberpunk 2077"
    print(f"Scoring reviews for: {product}\n")
    print("=" * 60)

    scored = score_reviews_for_product(product)

    for item in scored:
        label = "LOW" if item["suspicion_score"] < 30 else \
                "MEDIUM" if item["suspicion_score"] < 60 else "HIGH"
        print(f"\n[{item['review_id']}]  rating={item['rating']}  "
              f"suspicion={item['suspicion_score']}/100  [{label}]")
        print(f"  Text   : \"{item['review_text']}\"")
        print(f"  Reasons:")
        for reason in item["reasons"]:
            print(f"    - {reason}")

    print("\n" + "=" * 60)
    print("Done.")

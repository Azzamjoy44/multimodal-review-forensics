# Architecture

A FastAPI backend with a single-file vanilla-JS frontend. Every ML model is
trained offline and pre-loaded at server startup, so scoring at request time is
a single forward pass. This document maps the request flow, the scoring
pipeline, and where each model lives. For model designs and evaluation numbers
see [THESIS_NOTES.md](THESIS_NOTES.md); for datasets see [DATA.md](DATA.md).

## Request flow

| Endpoint | Loader | Scorer | Response |
|---|---|---|---|
| `GET /reviews?product_id=X&task=fake` | `load_amazon_reviews.py` | unified detector → `score_review_list` | Amazon fake-detection cards |
| `GET /amazon-fake-labeled` | `load_amazon_labeled.py` | unified detector | labelled (ground-truth) Amazon set |
| `GET /user-history?user_id=X` | `load_amazon_reviews.py` | DistilBERT sentiment + fake | reviewer dashboard |
| `GET /yelp-reviews?business_id=X` | `load_yelp_reviews.py` | Model B (or text-only baseline) | Yelp fake-detection cards |
| `GET /yelp-user-history?user_id=X` | `load_yelp_reviews.py` | Model B DistilBERT | Yelp reviewer history |
| `GET /gnn-graphs` | static ring PNGs | — | GNN Rings section |
| `POST /analyze` | review gate → merge fake+sentiment | unified detector + sentiment | Analyze box (+ saved to My Uploads) |
| `GET /personal-results` | `auth.get_uploads` | — | My Uploads (persisted per user) |

## Startup pre-loading (`main.py` lifespan)

A boot routine loads everything in parallel before the first request: the Amazon
and Yelp review indexes, the sklearn sentiment models, the Yelp multimodal
models, the LSTM/DistilBERT ONNX models, the review/non-review gate, and the
**unified fake detector** (CG + human-deception, served everywhere a piece of
Amazon review text needs an authenticity verdict). Deep models run through ONNX
Runtime (no deep-learning framework at serving time); classical models load from
joblib.

## Central scoring — `score_reviews.py`

- **Amazon** — `score_review_list(reviews, task, fake_override)`: rule-based
  first pass → fake votes (from the unified detector, passed in as
  `fake_override`) → sentiment votes → phrase/sentence highlight lists. One task
  (fake **or** sentiment) per call; the Analyze layer in `main.py` merges the two.
- **Yelp** — `score_yelp_review_list(reviews, variant, fake_override)`: serves
  Model B (text + 17 behavioural features) by default; the frontend "Detector"
  dropdown can switch to the text-only baseline or the focal/contrastive variants.

Each review comes back in a uniform shape: id, text, rating, rule-based score,
per-model fake votes, per-model sentiment votes (where relevant), and four
highlight lists (suspicious/genuine words, suspicious/genuine sentences). sklearn
models give word-level highlights (TF-IDF attribution); BiLSTM/DistilBERT give
sentence-level highlights.

## Model map

| Path (`data/`) | Task | Format |
|---|---|---|
| `combined_fake_models/`, `combined_fake_lstm_onnx/`, `combined_fake_distilbert_onnx/` | unified fake detector (CG + human deception) | joblib + ONNX |
| `yelp_multimodal_full_sklearn_models/`, `yelp_multimodal_lstm_full_onnx/`, `yelp_multimodal_distilbert_full_onnx/` | Model B (multi-modal Yelp) | joblib + ONNX |
| `sentiment_*.joblib`, `sentiment_lstm.onnx`, `distilbert_sentiment_onnx/` | sentiment | joblib + ONNX |
| `review_detector_sklearn_models/`, `review_detector_lstm_onnx/`, `review_detector_distilbert_onnx/` | review/non-review gate | joblib + ONNX |
| `yelp_fake_sklearn_models/`, `yelp_fake_{lstm,distilbert}_onnx/` (+ focal/contrastive) | Yelp text-only baseline & ablations | joblib + ONNX |

Each detector is a ten-model ensemble: eight sklearn pipelines (KNN, decision
tree, random forest, logistic regression, multinomial naive Bayes, AdaBoost,
XGBoost, MLP) on TF-IDF, plus a BiLSTM (GloVe) and a fine-tuned DistilBERT. The
verdict is a majority vote.

## Frontend (`static/index.html`)

Single file, no build step. Fixed sidebar with four sections: **Fake Review
Detection**, **Sentiment Analysis**, **Dashboard**, **GNN Rings**. Review cards
show a verdict badge, a confidence gauge, and a per-model dot strip; clicking a
dot reveals that model's word/sentence highlights. The Dashboard's **Analyze**
box accepts pasted text, a CSV, or screenshots (OCR via the Pixtral vision
model), runs the review gate, and saves results to the user's **My Uploads**.

## Persistence & auth (`auth.py`)

Local SQLite (`data/users.db`): SHA-256 password hashing, in-memory session
tokens. The **My Uploads** tables (`analyses`, `csv_uploads`) are keyed by
username, so a user's saved Analyze results survive server restarts.

## Training & evaluation

Classical models and all evaluation run locally (`train_*.py`, `evaluate_*.py`);
the GPU models (BiLSTM, DistilBERT, GNN) train in the Colab notebooks
(`train_*.ipynb`) and are exported to ONNX for serving. See
[THESIS_NOTES.md](THESIS_NOTES.md) for the full recipes and results.

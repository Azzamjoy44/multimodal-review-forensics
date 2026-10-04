"""
retrain_yelp_mlp.py
--------------------
Retrains the Yelp MLP model locally from yelp_split.csv using the same
hyperparameters as the Colab notebook, then saves a NumPy-compatible
yelp_fake_mlp.joblib and updates yelp_fake_thresholds.json.

Run once:
    python retrain_yelp_mlp.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
SKLEARN_DIR = os.path.join(DATA_DIR, "yelp_fake_sklearn_models")
SPLIT_CSV   = os.path.join(DATA_DIR, "yelp_split.csv")
SEED        = 42

print("Loading split...")
df      = pd.read_csv(SPLIT_CSV)
X_tr    = df[df["split"] == "train"]["text"].tolist()
y_tr    = df[df["split"] == "train"]["label"].tolist()
X_val   = df[df["split"] == "val"]["text"].tolist()
y_val   = df[df["split"] == "val"]["label"].tolist()
X_te    = df[df["split"] == "test"]["text"].tolist()
y_te    = df[df["split"] == "test"]["label"].tolist()
print(f"  Train={len(X_tr):,}  Val={len(X_val):,}  Test={len(X_te):,}")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2),
        stop_words="english", sublinear_tf=True,
    )),
    ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=SEED)),
])

print("\nTraining MLP...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pipeline.fit(X_tr, y_tr)

print("Tuning threshold on val set...")
val_probs = pipeline.predict_proba(X_val)[:, 1]
best_thresh, best_f1 = 0.5, 0.0
for t in np.arange(0.10, 0.90, 0.01):
    f = f1_score(y_val, (val_probs >= t).astype(int), average="macro", zero_division=0)
    if f > best_f1:
        best_f1, best_thresh = f, round(float(t), 2)
print(f"  Best threshold: {best_thresh:.2f}  (Val macro F1: {best_f1:.1%})")

print("Evaluating on test set...")
test_probs = pipeline.predict_proba(X_te)[:, 1]
y_pred = (test_probs >= best_thresh).astype(int)
print(f"  Accuracy : {accuracy_score(y_te, y_pred):.1%}")
print(f"  Macro F1 : {f1_score(y_te, y_pred, average='macro', zero_division=0):.1%}")
print(f"  Fake  P/R: {precision_score(y_te, y_pred, zero_division=0):.1%} / {recall_score(y_te, y_pred, zero_division=0):.1%}")

out_path = os.path.join(SKLEARN_DIR, "yelp_fake_mlp.joblib")
joblib.dump(pipeline, out_path)
print(f"\nSaved: {out_path}")

thresh_path = os.path.join(SKLEARN_DIR, "yelp_fake_thresholds.json")
with open(thresh_path) as f:
    thresholds = json.load(f)
thresholds["mlp"] = best_thresh
with open(thresh_path, "w") as f:
    json.dump(thresholds, f, indent=2)
print(f"Updated threshold in: {thresh_path}")

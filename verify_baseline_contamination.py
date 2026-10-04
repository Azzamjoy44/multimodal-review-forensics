"""
verify_baseline_contamination.py — is stale-split contamination the ONLY reason text-only
beats Model B on the frontend? Checks (ALL 10 models incl. KNN, matching the frontend):
  A. is Model B itself contaminated? (Model B train texts vs frontend test texts)
  B. do ALL baseline sklearn inflate? (served old-split vs FRESH current-split, per model)
  C. honest head-to-head: fresh text-only ensemble vs Model B ensemble on the frontend
RUN: PYTHONIOENCODING=utf-8 python verify_baseline_contamination.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from build_yelp_multimodal_features import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError:
    HAS_XGB = False

FEAT = ["rating","is_extreme","review_word_len","review_char_len","rating_dev_from_biz",
        "abs_rating_dev_from_biz","user_review_count","user_avg_rating","user_rating_std",
        "user_frac_positive","user_frac_extreme","user_is_singleton","user_max_reviews_per_day",
        "user_reviews_per_day","biz_review_count","biz_avg_rating","biz_rating_std"]
SK = ["knn","rf","dt","lr","nb","xgb","adaboost","mlp"]
KNN_SAMPLE = 20000


def mf1(y, p): return f1_score(y, p, average="macro", zero_division=0)


def model_list():
    m = [("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute"), 5000),
         ("rf", RandomForestClassifier(n_estimators=100, max_depth=40, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1), 5000),
         ("dt", DecisionTreeClassifier(class_weight="balanced", random_state=42), 5000),
         ("lr", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42), 10000),
         ("nb", MultinomialNB(), 10000),
         ("adaboost", AdaBoostClassifier(n_estimators=50, random_state=42), 5000),
         ("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=40, early_stopping=True, random_state=42), 10000)]
    if HAS_XGB:
        m.insert(5, ("xgb", XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=-1), 5000))
    return m


def main():
    sp = pd.read_csv("data/yelp_split.csv"); sp["text"] = sp["text"].astype(str)
    tr, va, te = (sp[sp.split == s].reset_index(drop=True) for s in ("train","val","test"))
    fe = pd.read_csv("data/yelp_frontend_multimodal.csv"); fe["review_text"] = fe["review_text"].astype(str)
    dl = pd.read_csv("dl_predictions.csv")
    y = fe["ground_truth_label"].values.astype(int)
    fe_norm = fe["review_text"].map(normalize)
    texts = fe["review_text"].tolist()

    # ---- CHECK A ----
    full = pd.read_csv("data/yelp_multimodal_features_full.csv", usecols=["text","split"])
    full["n"] = full["text"].astype(str).map(normalize)
    mb_tr = set(full[full.split == "train"]["n"]); base_tr = set(tr.text.map(normalize))
    ovl_mb   = fe_norm.map(lambda x: x in mb_tr).mean()
    ovl_base = fe_norm.map(lambda x: x in base_tr).mean()
    print("CHECK A - does either model's TRAIN overlap the frontend TEST (text leak)?")
    print(f"  Model B train vs frontend test overlap: {ovl_mb:.2%}   [{'OK disjoint' if ovl_mb<0.001 else 'LEAK'}]")
    print(f"  fresh-baseline train vs frontend test:  {ovl_base:.2%}   [{'OK disjoint' if ovl_base<0.001 else 'LEAK'}]")
    print("  (the SERVED baseline used an older/different split -> that's the contamination)\n")

    # ---- CHECK B: served vs fresh, per model (cache served preds for CHECK C) ----
    served_thr = json.load(open("data/yelp_fake_sklearn_models/yelp_fake_thresholds.json"))
    print("CHECK B - served (old-split, contaminated) vs FRESH (current-split, honest) baseline sklearn:")
    print(f"  {'model':<10}{'served':>9}{'fresh':>9}{'inflation':>11}")
    served_pred, fresh_pred = {}, {}
    for name, clf, maxf in model_list():
        sv = joblib.load(f"data/yelp_fake_sklearn_models/yelp_fake_{name}.joblib")
        served_pred[name] = (sv.predict_proba(texts)[:, 1] >= served_thr[name]).astype(int)
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=maxf, ngram_range=(1,2), sublinear_tf=True, min_df=2)), ("clf", clf)])
        if name == "knn":
            ix = np.random.RandomState(42).choice(len(tr), min(KNN_SAMPLE, len(tr)), replace=False)
            pipe.fit(tr.text.iloc[ix], tr.label.values[ix])
        else:
            pipe.fit(tr.text, tr.label.values)
        vp = pipe.predict_proba(va.text)[:, 1]; bt, bf = 0.5, -1
        for t in np.linspace(0.1, 0.9, 17):
            f = mf1(va.label.values, (vp >= t).astype(int))
            if f > bf: bf, bt = f, t
        fresh_pred[name] = (pipe.predict_proba(texts)[:, 1] >= bt).astype(int)
        sf, ff = mf1(y, served_pred[name]), mf1(y, fresh_pred[name])
        print(f"  {name:<10}{sf:>8.1%}{ff:>9.1%}{(sf-ff)*100:>+10.1f}", flush=True)

    # ---- CHECK C: ensembles (all 10 models, like the frontend) ----
    mm = pd.DataFrame({"text": texts})
    for c in FEAT: mm[c] = pd.to_numeric(fe[c], errors="coerce").fillna(0).astype("float32")
    mb_thr = json.load(open("data/yelp_multimodal_full_sklearn_models/thresholds.json"))
    mb_pred = {n: (joblib.load(f"data/yelp_multimodal_full_sklearn_models/{n}.joblib").predict_proba(mm)[:, 1] >= mb_thr.get(n, 0.5)).astype(int) for n in SK}

    def maj(d):
        M = np.vstack([np.asarray(d[k], int) for k in d]); return (M.sum(0) > M.shape[0]/2).astype(int)
    served_text = {**served_pred, "lstm": dl.base_lstm.values, "distilbert": dl.base_db.values}
    honest_text = {**fresh_pred,  "lstm": dl.base_lstm.values, "distilbert": dl.base_db.values}
    modelb      = {**mb_pred,     "lstm": dl.new_lstm.values,  "distilbert": dl.new_db.values}
    print("\nCHECK C - ENSEMBLE macro-F1 on the frontend (all 10 models, majority):")
    print(f"  text-only  SERVED (contaminated): {mf1(y, maj(served_text)):.1%}")
    print(f"  text-only  HONEST (fresh split) : {mf1(y, maj(honest_text)):.1%}")
    print(f"  Model B    (pure-full)          : {mf1(y, maj(modelb)):.1%}")
    print(f"\n  leakage-free DL-only:  text-only {mf1(y, maj({'lstm':dl.base_lstm.values,'distilbert':dl.base_db.values})):.1%}"
          f"  vs  Model B {mf1(y, maj({'lstm':dl.new_lstm.values,'distilbert':dl.new_db.values})):.1%}")


if __name__ == "__main__":
    main()

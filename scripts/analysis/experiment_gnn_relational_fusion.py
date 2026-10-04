# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
experiment_gnn_relational_fusion.py — the bigger swing: fuse the GNN's RELATIONAL features
(not just its score) into Model B and see how close we get to the 81% oracle.

Leakage control:
  * relational features used = STRUCTURAL (degree/comp/core/clustering/temporal, label-free) +
    neighbor_fraud_rate (neighbors' TRAIN labels, no self-leak). reviewer_train_fraud DROPPED (self-leak).
    -> safe to add to TRAIN features directly (feature-level fusion).
  * the GNN's own prob is in-sample on TRAIN -> NOT a train feature; stacked via a meta-LR on VAL.

Variants (train on TRAIN, threshold on VAL, eval TEST):
  A Model B           : text + 17 behavioral
  B + relational      : text + 17 behavioral + 6 relational (feature-level fusion)
  C + relational+GNN  : B's prob stacked with GNN prob_co_temporal (meta-LR on val)
Compare to oracle accuracy 81.1%.
RUN: PYTHONIOENCODING=utf-8 python experiment_gnn_relational_fusion.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
try:
    from xgboost import XGBClassifier; HAS_XGB = True
except ImportError:
    HAS_XGB = False

BEH = ["rating","is_extreme","review_word_len","review_char_len","rating_dev_from_biz",
       "abs_rating_dev_from_biz","user_review_count","user_avg_rating","user_rating_std",
       "user_frac_positive","user_frac_extreme","user_is_singleton","user_max_reviews_per_day",
       "user_reviews_per_day","biz_review_count","biz_avg_rating","biz_rating_std"]
REL = ["rel_degree","rel_comp_size","rel_core","rel_clustering","rel_temporal_degree","rel_neighbor_fraud_rate"]


def best_thr(y, p):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > bf: bf, bt = f, t
    return bt


def make(cols):
    ct = ColumnTransformer([("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1,2), sublinear_tf=True,
                                                       min_df=2, stop_words="english"), "text"),
                            ("num", MinMaxScaler(), cols)])
    if HAS_XGB:
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                            tree_method="hist", n_jobs=-1, random_state=42)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    return Pipeline([("feat", ct), ("clf", clf)])


def main():
    print("loading data + relational + GNN ...", flush=True)
    df = pd.read_csv("data/yelp_multimodal_features_full.csv", usecols=["text","label","split"] + BEH)
    df["text"] = df["text"].astype(str)
    R = np.load("data/yelp_graph/relational_features.npz", allow_pickle=True)
    relcols = [str(c) for c in R["cols"]]; rel = R["rel"]
    for j, c in enumerate(relcols):
        if c in REL: df[c] = rel[:, j]
    g = np.load("data/yelp_graph/yelp_gnn_temporal_preds.npz")
    df["gnn"] = g["prob_co_temporal"]
    df.columns = [str(c) for c in df.columns]      # ensure pure python str (npz names were numpy.str_)

    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train","val","test"))
    yv, yt = va.label.values.astype(int), te.label.values.astype(int)
    print(f"train {len(tr):,} / val {len(va):,} / test {len(te):,}\n", flush=True)

    def fit_eval(cols, name):
        p = make(cols); p.fit(tr, tr.label.values)
        pv, pt = p.predict_proba(va)[:, 1], p.predict_proba(te)[:, 1]
        thr = best_thr(yv, pv)
        f = f1_score(yt, (pt >= thr).astype(int), average="macro")
        print(f"  {name:<28} test macroF1 {f:.1%}", flush=True)
        return pv, pt, f

    pv_a, pt_a, f_a = fit_eval(BEH, "A: Model B (text+beh)")
    pv_b, pt_b, f_b = fit_eval(BEH + REL, "B: + relational")

    # C: stack B's prob with the held-out GNN prob (meta-LR on val)
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta.fit(np.c_[pv_b, va.gnn.values], yv)
    cv = meta.predict_proba(np.c_[pv_b, va.gnn.values])[:, 1]
    ct_ = meta.predict_proba(np.c_[pt_b, te.gnn.values])[:, 1]
    f_c = f1_score(yt, (ct_ >= best_thr(yv, cv)).astype(int), average="macro")
    print(f"  {'C: + relational + GNN-stack':<28} test macroF1 {f_c:.1%}", flush=True)

    print("\n" + "=" * 56)
    print("RELATIONAL FUSION vs the oracle")
    print("=" * 56)
    print(f"  A Model B (text+behavioral)     : {f_a:.1%}")
    print(f"  B + relational features         : {f_b:.1%}   ({(f_b-f_a)*100:+.1f} pp)")
    print(f"  C + relational + GNN-prob stack : {f_c:.1%}   ({(f_c-f_a)*100:+.1f} pp)")
    print(f"  ---")
    print(f"  prior score-only fusion         : 76.9%")
    print(f"  ORACLE ceiling (accuracy)       : 81.1%")


if __name__ == "__main__":
    main()

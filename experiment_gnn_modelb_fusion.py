"""
experiment_gnn_modelb_fusion.py — does fusing the GNN (relational) with Model B (text+behavioral)
help? Pre-flight: complementarity + oracle upper bound, then a leakage-free stacking fusion.

Alignment: GNN temporal preds row i == yelp_multimodal_features_full.csv row i (verified 100%
label match). GNN val/test are held-out (GNN never trained on those labels), so a meta-model
trained on VAL and evaluated on TEST is leakage-free. Model B here = its sklearn ensemble
(fast; representative ~76-77%); the GNN signal = prob_co_temporal.

RUN: PYTHONIOENCODING=utf-8 python experiment_gnn_modelb_fusion.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

FEAT = ["rating","is_extreme","review_word_len","review_char_len","rating_dev_from_biz",
        "abs_rating_dev_from_biz","user_review_count","user_avg_rating","user_rating_std",
        "user_frac_positive","user_frac_extreme","user_is_singleton","user_max_reviews_per_day",
        "user_reviews_per_day","biz_review_count","biz_avg_rating","biz_rating_std"]
SK = ["rf","dt","lr","nb","xgb","adaboost","mlp"]   # Model B sklearn (no knn, for speed)
MBDIR = "data/yelp_multimodal_full_sklearn_models"


def best_thr(y, p):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > bf: bf, bt = f, t
    return bt


def main():
    print("loading features_full + GNN preds ...", flush=True)
    ff = pd.read_csv("data/yelp_multimodal_features_full.csv", usecols=["text","label","split"] + FEAT)
    ff["text"] = ff["text"].astype(str)
    g = np.load("data/yelp_graph/yelp_gnn_temporal_preds.npz")
    assert len(g["label"]) == len(ff) and (g["label"].astype(int) == ff["label"].values).all(), "alignment broke"
    ff["gnn"] = g["prob_co_temporal"]

    va = ff[ff.split == "val"].reset_index(drop=True)
    te = ff[ff.split == "test"].reset_index(drop=True)
    yv, yt = va.label.values.astype(int), te.label.values.astype(int)
    print(f"val {len(va):,} / test {len(te):,}\n", flush=True)

    def mm(df):
        d = pd.DataFrame({"text": df.text.values})
        for c in FEAT: d[c] = df[c].astype("float32")
        return d
    mbthr = json.load(open(f"{MBDIR}/thresholds.json"))

    # Model B sklearn: ensemble vote-fraction (held-out: trained on TRAIN, predicting val/test)
    def mb_scores(df):
        X = mm(df); votes = np.zeros(len(df))
        for n in SK:
            p = joblib.load(f"{MBDIR}/{n}.joblib").predict_proba(X)[:, 1]
            votes += (p >= mbthr.get(n, 0.5)).astype(int)
        return votes / len(SK)                 # fraction of MB sklearn voting fake
    mb_va, mb_te = mb_scores(va), mb_scores(te)
    print("scored Model B sklearn on val+test", flush=True)

    # ---- individual detectors (test), thresholds tuned on val ----
    mb_thr = best_thr(yv, mb_va); mb_pred = (mb_te >= mb_thr).astype(int)
    gn_thr = best_thr(yv, va.gnn.values); gn_pred = (te.gnn.values >= gn_thr).astype(int)
    f_mb, f_gn = f1_score(yt, mb_pred, average="macro"), f1_score(yt, gn_pred, average="macro")

    # ---- complementarity / oracle upper bound ----
    mb_ok, gn_ok = (mb_pred == yt), (gn_pred == yt)
    both = (mb_ok & gn_ok).mean(); mb_only = (mb_ok & ~gn_ok).mean()
    gn_only = (~mb_ok & gn_ok).mean(); neither = (~mb_ok & ~gn_ok).mean()
    oracle_acc = (mb_ok | gn_ok).mean()
    mb_wrong_gn_right = (gn_ok[~mb_ok]).mean() if (~mb_ok).any() else 0

    # ---- leakage-free stacking: meta-LR on VAL [mb, gnn] -> eval TEST ----
    meta = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta.fit(np.c_[mb_va, va.gnn.values], yv)
    fus_p = meta.predict_proba(np.c_[mb_te, te.gnn.values])[:, 1]
    ft = best_thr(yv, meta.predict_proba(np.c_[mb_va, va.gnn.values])[:, 1])
    f_fus = f1_score(yt, (fus_p >= ft).astype(int), average="macro")

    print("\n" + "=" * 60)
    print(f"TEST (n={len(te):,})  macro-F1")
    print("=" * 60)
    print(f"  Model B (sklearn ensemble) : {f_mb:.1%}")
    print(f"  GNN (prob_co_temporal)     : {f_gn:.1%}")
    print(f"  FUSION (meta-LR, val->test): {f_fus:.1%}   ({(f_fus-max(f_mb,f_gn))*100:+.1f} pp vs best single)")
    print("\nComplementarity of their predictions:")
    print(f"  both correct {both:.1%} | MB-only {mb_only:.1%} | GNN-only {gn_only:.1%} | both wrong {neither:.1%}")
    print(f"  of reviews MB gets WRONG, GNN gets right: {mb_wrong_gn_right:.1%}")
    print(f"  ORACLE upper bound (right if EITHER right): {oracle_acc:.1%} accuracy")
    print("\nRead: if GNN-only + oracle are high -> complementary, fusion worth it; if ~0 -> redundant.")


if __name__ == "__main__":
    main()

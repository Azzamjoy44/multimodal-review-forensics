"""
model_b_member_stats.py — exact per-member + ensemble stats for the SERVED pure-full Model B
on the frontend Yelp test set (data/yelp_frontend_multimodal.csv, n=17,891, balanced 50/50).
8 multi-modal sklearn scored live; LSTM + DistilBERT taken from dl_predictions_all.csv (mb_lstm,
mb_db). Reports macro-F1 / fake recall / fake precision / genuine specificity for every member,
the 8-sklearn majority ensemble, and the full 10-model served ensemble.
RUN: PYTHONIOENCODING=utf-8 python -u model_b_member_stats.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score
warnings.filterwarnings("ignore")
import model_b

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def m(y, pred):
    return (round(f1_score(y, pred, average="macro") * 100, 1),
            round(recall_score(y, pred, pos_label=1) * 100, 1),
            round(precision_score(y, pred, pos_label=1) * 100, 1),
            round(recall_score(y, pred, pos_label=0) * 100, 1))


def main():
    df = pd.read_csv(os.path.join(DATA, "yelp_frontend_multimodal.csv"))
    df["review_text"] = df["review_text"].astype(str)
    y = df["ground_truth_label"].values.astype(int)
    mm = pd.DataFrame({"text": df["review_text"].tolist()})
    for c in model_b.FEAT_COLS:
        mm[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    dl = pd.read_csv(os.path.join(HERE, "dl_predictions_all.csv"))
    assert len(dl) == len(df) and (dl["ground_truth_label"].values == y).all()

    sk, thr = model_b._load_sklearn()
    rows, votes = [], []
    for n in model_b.SK_NAMES:                       # knn, rf, dt, lr, nb, xgb, adaboost, mlp
        pred = (sk[n].predict_proba(mm)[:, 1] >= thr[n]).astype(int)
        votes.append(pred); rows.append((n.upper(), *m(y, pred)))
        print(f"  {n} done", flush=True)
    for name, col in [("LSTM", "mb_lstm"), ("DistilBERT", "mb_db")]:
        rows.append((name, *m(y, dl[col].values.astype(int))))

    sk_M = np.vstack(votes)
    sk_ens = (sk_M.sum(0) > sk_M.shape[0] / 2).astype(int)
    rows.append(("8-sklearn ensemble", *m(y, sk_ens)))
    full_M = np.vstack([sk_M, dl["mb_lstm"].values, dl["mb_db"].values])
    full_ens = (full_M.sum(0) > full_M.shape[0] / 2).astype(int)
    rows.append(("FULL 10-model ensemble", *m(y, full_ens)))

    print(f"\nServed pure-full Model B — n={len(df)} (fake rate {y.mean():.3f})")
    print(f"{'member':<24}{'macroF1':>9}{'fakeR':>8}{'fakeP':>8}{'genSpec':>9}")
    for nm, f1, fr, fp, gs in rows:
        print(f"{nm:<24}{f1:>8.1f}%{fr:>7.1f}%{fp:>7.1f}%{gs:>8.1f}%")
    pd.DataFrame(rows, columns=["member", "macroF1", "fakeR", "fakeP", "genSpec"]).to_csv(
        os.path.join(HERE, "model_b_member_stats.csv"), index=False)
    print("\nwrote model_b_member_stats.csv")


if __name__ == "__main__":
    main()

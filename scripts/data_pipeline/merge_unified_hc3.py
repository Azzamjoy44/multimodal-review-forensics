# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
merge_unified_hc3.py — combine the LOCAL 8 unified sklearn with the Colab-computed DL predictions
(unified_hc3_dl_preds.csv) into the served detector's ZERO-SHOT OOD result on HC3 (8,000 balanced
ChatGPT-vs-human, never trained on). Reports macro-F1 / AI(ChatGPT)-recall / human-specificity per
member + the 8-sklearn majority + the full 10-model ensemble — the defensible OOD number.
RUN: PYTHONIOENCODING=utf-8 python -u merge_unified_hc3.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
import model_unified as mu

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
DL_PREDS = os.path.join(HERE, "unified_hc3_dl_preds.csv")


def stats(y, pred):
    ai = y == 1; hu = y == 0
    return (round(f1_score(y, pred, average="macro") * 100, 1),
            round(pred[ai].mean() * 100, 1),            # AI / ChatGPT recall
            round((pred[hu] == 0).mean() * 100, 1))     # human specificity


def main():
    if not os.path.exists(DL_PREDS):
        raise SystemExit("unified_hc3_dl_preds.csv not found — run evaluate_unified_hc3_colab.ipynb on Colab first.")
    df = pd.read_csv(os.path.join(DATA, "ood_hc3_full.csv")); df["review_text"] = df["review_text"].astype(str)
    texts = df["review_text"].tolist(); y = df["ground_truth_label"].values.astype(int)
    dl = pd.read_csv(DL_PREDS)
    assert len(dl) == len(df) and (dl["ground_truth_label"].values == y).all(), \
        "unified_hc3_dl_preds.csv not row-aligned with ood_hc3_full.csv"

    sk, thr = mu._load_sklearn()
    votes = {}
    for n in mu.SK_NAMES:
        votes[n] = (sk[n].predict_proba(texts)[:, 1] >= thr[n]).astype(int)
        print(f"  sklearn {n} done", flush=True)
    votes["lstm"] = dl["lstm_pred"].values.astype(int)
    votes["distilbert"] = dl["db_pred"].values.astype(int)

    order = ["distilbert", "lstm", "mlp", "lr", "xgb", "knn", "rf", "nb", "dt", "adaboost"]
    rows = [(nm, *stats(y, votes[nm])) for nm in order]
    sk8 = np.vstack([votes[n] for n in mu.SK_NAMES])
    rows.append(("sklearn majority (8)", *stats(y, (sk8.sum(0) > 4).astype(int))))
    all10 = np.vstack([votes[n] for n in mu.SK_NAMES] + [votes["lstm"], votes["distilbert"]])
    rows.append(("FULL 10-model ensemble", *stats(y, (all10.sum(0) > 5).astype(int))))

    print(f"\nUnified detector — ZERO-SHOT OOD on HC3 (n={len(df)}, balanced ChatGPT vs human)")
    print(f"{'model':<24}{'macroF1':>9}{'AI-recall':>11}{'human-spec':>12}")
    for nm, f1, air, hs in rows:
        print(f"{nm:<24}{f1:>8.1f}%{air:>10.1f}%{hs:>11.1f}%")
    pd.DataFrame(rows, columns=["model", "macroF1", "ai_recall", "human_spec"]).to_csv(
        os.path.join(HERE, "unified_hc3_ood_stats.csv"), index=False)
    print("\nwrote unified_hc3_ood_stats.csv")


if __name__ == "__main__":
    main()

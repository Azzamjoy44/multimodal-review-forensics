# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
merge_unified_ensemble.py — combine the LOCAL 8 sklearn (fast) with the Colab-computed DL
predictions (unified_dl_preds.csv from compute_unified_dl_preds_colab.ipynb) into the full
unified-detector stats on the combined test (n=13,179). Fills the missing 10-model-ensemble row.
Reports macro-F1 / CG-fake recall / human-fake recall / genuine specificity per member + the
8-sklearn majority + the 10-model majority. RUN: PYTHONIOENCODING=utf-8 python -u merge_unified_ensemble.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import warnings
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
import model_unified as mu

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
DL_PREDS = os.path.join(HERE, "unified_dl_preds.csv")


def stats(y, ft, pred):
    g = ft == "genuine"; cg = ft == "cg_fake"; hu = ft == "human_fake"
    return (round(f1_score(y, pred, average="macro") * 100, 1),
            round(pred[cg].mean() * 100, 1),
            round(pred[hu].mean() * 100, 1),
            round((pred[g] == 0).mean() * 100, 1))


def main():
    if not os.path.exists(DL_PREDS):
        raise SystemExit("unified_dl_preds.csv not found — run compute_unified_dl_preds_colab.ipynb "
                         "on Colab first and drop the CSV in the project root.")
    df = pd.read_csv(os.path.join(DATA, "combined_fake_reviews.csv"))
    df = df[df.split == "test"].reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    texts = df["text"].tolist(); y = df["label"].values.astype(int); ft = df["fake_type"].values

    dl = pd.read_csv(DL_PREDS)
    assert len(dl) == len(df) and (dl["label"].values == y).all() and (dl["fake_type"].values == ft).all(), \
        "unified_dl_preds.csv is not row-aligned with the combined test split"

    sk, thr = mu._load_sklearn()
    votes = {}
    for n in mu.SK_NAMES:
        votes[n] = (sk[n].predict_proba(texts)[:, 1] >= thr[n]).astype(int)
        print(f"  sklearn {n} done", flush=True)
    votes["lstm"] = dl["lstm_pred"].values.astype(int)
    votes["distilbert"] = dl["db_pred"].values.astype(int)

    order = ["distilbert", "lstm", "mlp", "lr", "xgb", "rf", "nb", "knn", "dt", "adaboost"]
    rows = [(nm, *stats(y, ft, votes[nm])) for nm in order]
    sk8 = np.vstack([votes[n] for n in mu.SK_NAMES])
    rows.append(("sklearn majority (8)", *stats(y, ft, (sk8.sum(0) > 4).astype(int))))
    all10 = np.vstack([votes[n] for n in mu.SK_NAMES] + [votes["lstm"], votes["distilbert"]])
    rows.append(("FULL 10-model ensemble", *stats(y, ft, (all10.sum(0) > 5).astype(int))))

    print(f"\nUnified detector — combined test n={len(df)} "
          f"(genuine {(ft=='genuine').sum()} / cg {(ft=='cg_fake').sum()} / human {(ft=='human_fake').sum()})")
    print(f"{'model':<24}{'macroF1':>9}{'CG-R':>7}{'human-R':>9}{'genSpec':>9}")
    for nm, f1, cgr, hur, gs in rows:
        print(f"{nm:<24}{f1:>8.1f}%{cgr:>6.1f}%{hur:>8.1f}%{gs:>8.1f}%")
    pd.DataFrame(rows, columns=["model", "macroF1", "cg_recall", "human_recall", "genuine_spec"]).to_csv(
        os.path.join(HERE, "unified_ensemble_stats.csv"), index=False)
    print("\nwrote unified_ensemble_stats.csv")


if __name__ == "__main__":
    main()

# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
human_fake_recall_ci.py — 95% Wilson confidence intervals for each unified-detector model's
HUMAN-FAKE recall (the scarce class, n=164 in the combined test). sklearn run locally; LSTM +
DistilBERT predictions read from unified_dl_preds.csv (Colab GPU). Wilson interval is the correct
small-sample CI for a proportion. RUN: PYTHONIOENCODING=utf-8 python -u human_fake_recall_ci.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import model_unified as mu

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - h), min(1.0, c + h)


def main():
    df = pd.read_csv(os.path.join(DATA, "combined_fake_reviews.csv"))
    df = df[df.split == "test"].reset_index(drop=True); df["text"] = df["text"].astype(str)
    texts = df["text"].tolist(); ft = df["fake_type"].values
    hu = ft == "human_fake"; n = int(hu.sum())
    dl = pd.read_csv(os.path.join(HERE, "unified_dl_preds.csv"))
    assert len(dl) == len(df) and (dl["fake_type"].values == ft).all()

    sk, thr = mu._load_sklearn()
    votes = {nm: (sk[nm].predict_proba(texts)[:, 1] >= thr[nm]).astype(int) for nm in mu.SK_NAMES}
    votes["lstm"] = dl["lstm_pred"].values.astype(int)
    votes["distilbert"] = dl["db_pred"].values.astype(int)
    sk8 = np.vstack([votes[nm] for nm in mu.SK_NAMES]); votes["sklearn majority (8)"] = (sk8.sum(0) > 4).astype(int)
    all10 = np.vstack([votes[nm] for nm in mu.SK_NAMES] + [votes["lstm"], votes["distilbert"]])
    votes["FULL 10-model ensemble"] = (all10.sum(0) > 5).astype(int)

    order = ["distilbert", "lstm", "mlp", "lr", "xgb", "knn", "rf", "nb", "dt", "adaboost",
             "sklearn majority (8)", "FULL 10-model ensemble"]
    print(f"Human-fake recall with 95% Wilson CI (n={n} human-fake reviews)\n")
    print(f"{'model':<24}{'caught':>9}{'recall':>9}{'95% CI':>16}")
    rows = []
    for nm in order:
        k = int(votes[nm][hu].sum())
        p, lo, hione = wilson(k, n)
        print(f"{nm:<24}{f'{k}/{n}':>9}{p*100:>8.1f}%   [{lo*100:>4.1f}, {hione*100:>4.1f}]")
        rows.append((nm, k, n, round(p*100, 1), round(lo*100, 1), round(hione*100, 1)))
    pd.DataFrame(rows, columns=["model", "caught", "n", "recall", "ci_low", "ci_high"]).to_csv(
        os.path.join(HERE, "human_fake_recall_ci.csv"), index=False)
    print("\nwrote human_fake_recall_ci.csv")


if __name__ == "__main__":
    main()

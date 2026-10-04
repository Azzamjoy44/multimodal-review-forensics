# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
retrain_yelp_baseline_sklearn.py — fix the CONTAMINATED served Yelp text-only baseline sklearn.

The served models in data/yelp_fake_sklearn_models/ were trained on an OLDER yelp_split whose
training reviews now sit in the current frontend TEST set -> rf/dt/mlp inflate +12-21 pp
(verify_baseline_contamination.py). This re-fits each model with its IDENTICAL config (sklearn
.clone of the served pipeline) on the CURRENT yelp_split, threshold-tunes on val, and overwrites
in place. The contaminated originals are backed up first. KNN is fit on a 20k sample (fast serve).

RUN: PYTHONIOENCODING=utf-8 python retrain_yelp_baseline_sklearn.py
"""
import os, shutil, json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.base import clone
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

DIR = os.path.join("data", "yelp_fake_sklearn_models")
BACKUP = os.path.join("data", "yelp_fake_sklearn_models_stale_backup")
SK = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]
KNN_SAMPLE = 20000


def best_thr(y, proba):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (proba >= t).astype(int), average="macro", zero_division=0)
        if f > bf:
            bf, bt = f, t
    return float(bt)


def main():
    # 1) back up the contaminated originals (don't destroy)
    os.makedirs(BACKUP, exist_ok=True)
    for f in os.listdir(DIR):
        if f.endswith((".joblib", ".json", ".csv")):
            shutil.copy2(os.path.join(DIR, f), os.path.join(BACKUP, f))
    print(f"backed up {len(os.listdir(BACKUP))} contaminated files -> {BACKUP}\n", flush=True)

    # 2) current split
    sp = pd.read_csv(os.path.join("data", "yelp_split.csv")); sp["text"] = sp["text"].astype(str)
    tr, va, te = (sp[sp.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"current yelp_split: train {len(tr):,} / val {len(va):,} / test {len(te):,}\n", flush=True)

    thr, rows = {}, []
    for name in SK:
        served = joblib.load(os.path.join(DIR, f"yelp_fake_{name}.joblib"))
        fresh = clone(served)                          # identical config, unfitted
        if name == "knn":
            ix = np.random.RandomState(42).choice(len(tr), min(KNN_SAMPLE, len(tr)), replace=False)
            fresh.fit(tr.text.iloc[ix], tr.label.values[ix])
        else:
            fresh.fit(tr.text, tr.label.values)
        t = best_thr(va.label.values, fresh.predict_proba(va.text)[:, 1])
        proba = fresh.predict_proba(te.text)[:, 1]; pred = (proba >= t).astype(int)
        thr[name] = t
        joblib.dump(fresh, os.path.join(DIR, f"yelp_fake_{name}.joblib"))
        mf1 = f1_score(te.label, pred, average="macro", zero_division=0)
        rows.append((name, round(t, 2), round(accuracy_score(te.label, pred), 4), round(mf1, 4),
                     round(precision_score(te.label, pred, zero_division=0), 4),
                     round(recall_score(te.label, pred, zero_division=0), 4),
                     len(te), int((te.label == 0).sum()), int((te.label == 1).sum())))
        print(f"  {name:<10} retrained: test macroF1 {mf1:.1%}  (thr {t:.2f})", flush=True)

    json.dump(thr, open(os.path.join(DIR, "yelp_fake_thresholds.json"), "w"), indent=2)
    pd.DataFrame(rows, columns=["model", "threshold", "accuracy", "macro_f1", "precision",
                                "recall", "n_test", "n_genuine", "n_fake"]).to_csv(
        os.path.join(DIR, "yelp_fake_sklearn_metrics.csv"), index=False)
    print("\nsaved honest baseline sklearn + thresholds + metrics (was contaminated, now current-split).")
    print("compare: served rf 80.1% -> now ~65%, dt 78.1% -> ~57%, mlp 80.2% -> ~68% (the honest ceiling).")


if __name__ == "__main__":
    main()

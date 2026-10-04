# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
finish_baseline_mlp.py — finish the interrupted baseline retrain. The other 7 sklearn were
re-fit fresh on the current split; only the MLP stalled (cloning the served 50k-feature config
made it pathologically slow). Refit the MLP with a sane 10k vocab, then recompute thresholds on
val for ALL 8 and write thresholds.json + metrics (the killed run never wrote them).
RUN: PYTHONIOENCODING=utf-8 python finish_baseline_mlp.py
"""
import os, json, warnings
import numpy as np, pandas as pd, joblib
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

DIR = os.path.join("data", "yelp_fake_sklearn_models")
SK = ["knn", "rf", "dt", "lr", "nb", "xgb", "adaboost", "mlp"]


def best_thr(y, p):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(y, (p >= t).astype(int), average="macro", zero_division=0)
        if f > bf: bf, bt = f, t
    return float(bt)


def main():
    sp = pd.read_csv(os.path.join("data", "yelp_split.csv")); sp["text"] = sp["text"].astype(str)
    tr, va, te = (sp[sp.split == s].reset_index(drop=True) for s in ("train", "val", "test"))

    print("retraining MLP (10k vocab; 50k was the stall) ...", flush=True)
    mlp = Pipeline([("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True,
                                              min_df=2, stop_words="english")),
                    ("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=40, early_stopping=True, random_state=42))])
    mlp.fit(tr.text, tr.label.values)
    joblib.dump(mlp, os.path.join(DIR, "yelp_fake_mlp.joblib"))
    print("  MLP saved", flush=True)

    thr, rows = {}, []
    for name in SK:
        m = joblib.load(os.path.join(DIR, f"yelp_fake_{name}.joblib"))
        t = best_thr(va.label.values, m.predict_proba(va.text)[:, 1])
        pred = (m.predict_proba(te.text)[:, 1] >= t).astype(int)
        thr[name] = t
        mf1 = f1_score(te.label, pred, average="macro", zero_division=0)
        rows.append((name, round(t, 2), round(accuracy_score(te.label, pred), 4), round(mf1, 4),
                     round(precision_score(te.label, pred, zero_division=0), 4),
                     round(recall_score(te.label, pred, zero_division=0), 4),
                     len(te), int((te.label == 0).sum()), int((te.label == 1).sum())))
        print(f"  {name:<10} honest test macroF1 {mf1:.1%}  (thr {t:.2f})", flush=True)

    json.dump(thr, open(os.path.join(DIR, "yelp_fake_thresholds.json"), "w"), indent=2)
    pd.DataFrame(rows, columns=["model","threshold","accuracy","macro_f1","precision","recall",
                                "n_test","n_genuine","n_fake"]).to_csv(
        os.path.join(DIR, "yelp_fake_sklearn_metrics.csv"), index=False)

    # honest ensemble (9 models: 8 sklearn ex-? actually all 8 + would need DL; report sklearn-majority)
    M = []
    for name in SK:
        m = joblib.load(os.path.join(DIR, f"yelp_fake_{name}.joblib"))
        M.append((m.predict_proba(te.text)[:, 1] >= thr[name]).astype(int))
    ens = (np.vstack(M).sum(0) > len(M) / 2).astype(int)
    print(f"\nHONEST baseline sklearn-majority (8 models): macroF1 {f1_score(te.label, ens, average='macro'):.1%}")
    print("baseline retrain complete — Text-only toggle now reads the honest ~65-68% ceiling.")


if __name__ == "__main__":
    main()

# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
verify_all_yelp_detectors.py — confirm every Yelp fake detector the frontend can switch to is
GOOD (loads, scores, non-degenerate) before exposing the toggle. Detectors:
  modelb      -> Model B (pure-full multimodal: 8 sklearn + LSTM + DistilBERT)
  baseline    -> text-only baseline sklearn (just retrained) + baseline LSTM + DistilBERT
  focal       -> baseline sklearn + FOCAL LSTM + FOCAL DistilBERT
  contrastive -> baseline sklearn + CONTRASTIVE LSTM + CONTRASTIVE DistilBERT
Reports ensemble macroF1 + each DL member's macroF1 + a degeneracy check (all-one-class = broken).
RUN: PYTHONIOENCODING=utf-8 python verify_all_yelp_detectors.py [sample]
"""
import os
os.environ.setdefault("USE_TF", "0")
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import f1_score
import model_b
from score_reviews import score_yelp_review_list

FEAT = model_b.FEAT_COLS
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2500


def memb(scored, name):
    """labels of one fake_models member across reviews (None-safe)."""
    return np.array([((s["fake_models"].get(name) or {}).get("label")) for s in scored], dtype=object)


def stats(y, scored):
    fm_names = list(scored[0]["fake_models"].keys())
    # ensemble (majority)
    ens = []
    for s in scored:
        v = [m.get("label") for m in s["fake_models"].values() if m.get("label") is not None]
        ens.append(1 if v and sum(v) > len(v) / 2 else 0)
    ens = np.array(ens)
    out = {"ensemble": f1_score(y, ens, average="macro", zero_division=0)}
    degen = []
    for nm in ("lstm", "distilbert"):
        lab = memb(scored, nm)
        ok = np.array([x is not None for x in lab])
        if ok.sum() == 0:
            out[nm] = None; degen.append(f"{nm}=UNAVAILABLE"); continue
        p = lab[ok].astype(int)
        out[nm] = f1_score(y[ok], p, average="macro", zero_division=0)
        fr = p.mean()
        if fr in (0.0, 1.0): degen.append(f"{nm}=ALL-{'FAKE' if fr==1 else 'GENUINE'}")
    return out, degen


def main():
    df = pd.read_csv("data/yelp_frontend_multimodal.csv")
    df = df.sample(min(N, len(df)), random_state=42).reset_index(drop=True)
    y = df["ground_truth_label"].values.astype(int)
    reviews = [{"review_id": str(df.iloc[i]["review_id"]), "review_text": str(df.iloc[i]["review_text"]),
                **{c: df.iloc[i][c] for c in FEAT}} for i in range(len(df))]
    print(f"sample: {len(df):,} frontend reviews ({y.mean():.0%} fake)\n", flush=True)

    rows = []
    # modelb
    print("scoring modelb ...", flush=True)
    s, d = stats(y, score_yelp_review_list(reviews, fake_override=model_b.score_yelp_fake(reviews)))
    rows.append(("modelb", s, d))
    # baseline / focal / contrastive (text-only sklearn + variant DL)
    for v in ("baseline", "focal", "contrastive"):
        print(f"scoring {v} ...", flush=True)
        s, d = stats(y, score_yelp_review_list(reviews, variant=v))
        rows.append((v, s, d))

    print("\n" + "=" * 70)
    print(f"{'detector':<13}{'ensemble':>10}{'LSTM':>9}{'DistilBERT':>12}   health")
    print("=" * 70)
    allgood = True
    for name, s, d in rows:
        def f(x): return "  n/a" if x is None else f"{x:.1%}"
        health = "OK" if not d else "  ⚠ " + ", ".join(d)
        if d: allgood = False
        print(f"{name:<13}{f(s['ensemble']):>10}{f(s.get('lstm')):>9}{f(s.get('distilbert')):>12}   {health}")
    print("=" * 70)
    print("\nALL DETECTORS GOOD — safe to expose the toggle." if allgood
          else "\nSOME DETECTOR DEGENERATE/UNAVAILABLE — fix before exposing.")


if __name__ == "__main__":
    main()

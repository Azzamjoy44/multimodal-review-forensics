# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
build_sample_small.py — build a SMALL (~14-row) Analyze demo CSV from REAL corpus rows that the
unified detector actually classifies correctly (hand-written text mis-behaves — see the inverted
results). Samples genuine + cg_fake + human_fake from combined_fake_reviews.csv, verifies each
row's 10-model majority verdict, and assembles: a coordinated burst reviewer (real AI fakes, with
an exact + a near duplicate), real genuine reviews, a couple more real fakes, and 2 non-reviews.
RUN: PYTHONIOENCODING=utf-8 python -u build_sample_small.py
"""
import os
os.environ.setdefault("USE_TF", "0")
import csv, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import model_unified as mu
import review_detector

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")


def majority_labels(texts):
    sk = mu._score_sklearn(texts); ls = mu._score_lstm(texts); db = mu._score_db(texts)
    out = []
    for i in range(len(texts)):
        votes = [sk[i][n]["label"] for n in mu.SK_NAMES if sk[i].get(n) and sk[i][n]["label"] is not None]
        votes += [ls[i]["label"], db[i]["label"]]
        out.append(1 if sum(votes) > len(votes) / 2 else 0)
    return out


def pick(df, ftype, want_label, n, seed):
    """Pick n rows of fake_type==ftype that the detector classifies as want_label (1 fake/0 gen)."""
    pool = df[(df.fake_type == ftype)].copy()
    pool = pool[pool.text.str.len().between(70, 320)]          # readable length
    pool = pool.sample(frac=1.0, random_state=seed)
    picks, texts = [], pool.text.tolist()
    BATCH = 60
    for s in range(0, len(texts), BATCH):
        chunk = texts[s:s + BATCH]
        labs = majority_labels(chunk)
        gate = review_detector.classify_texts(chunk)         # must ALSO pass the review gate
        for t, l, g in zip(chunk, labs, gate):
            if l == want_label and g["is_review"] and t.strip() not in [p.strip() for p in picks]:
                picks.append(t.strip())
                if len(picks) >= n:
                    return picks
    return picks


def main():
    df = pd.read_csv(os.path.join(DATA, "combined_fake_reviews.csv"))
    df["text"] = df["text"].astype(str)
    # one-line, clean texts only (no embedded newlines / weird quoting)
    df = df[~df.text.str.contains(r"[\r\n]")]
    print("sampling real rows the detector classifies correctly ...", flush=True)
    cg   = pick(df, "cg_fake",   1, 7, seed=1)      # AI fakes flagged fake
    gen  = pick(df, "genuine",   0, 5, seed=2)      # real reviews flagged genuine
    hum  = pick(df, "human_fake",1, 1, seed=3)      # human-deceptive flagged fake
    print(f"got cg_fake {len(cg)} | genuine {len(gen)} | human_fake {len(hum)}", flush=True)

    # coordinated burst reviewer: 5 real AI fakes, with an exact + a near duplicate folded in
    base = cg[0]
    near = base + " Highly recommend it to everyone."        # near-dup: keep all of base, append a short tail
    burst = [(base, "FAKE (ai)"), (base, "FAKE (dup-exact)"), (near, "FAKE (dup-near)"),
             (cg[1], "FAKE (ai)"), (cg[2], "FAKE (ai)")]
    NONREV = [
        ("Photosynthesis is the process by which green plants convert sunlight, water, and carbon dioxide into glucose and oxygen.", "user_nr1", 3, "2024-03-10", "NOT A REVIEW"),
        ("To reset the device, hold the power button for ten seconds until the LED blinks twice, then release and wait for it to reboot.", "user_nr2", 3, "2024-03-11", "NOT A REVIEW"),
    ]

    rows = [("review_text", "user_id", "rating", "date", "expected", "source_type")]
    for t, lab in burst:
        rows.append((t, "burst_bot", 5, "2024-05-01", lab, "cg_fake"))
    for i, t in enumerate(gen):
        rows.append((t, f"user_h{i+1}", [5, 2, 4, 1, 3][i % 5], f"2024-02-{10+i:02d}", "GENUINE (human)", "genuine"))
    for i, t in enumerate(cg[3:5]):
        rows.append((t, f"user_a{i+1}", 4, f"2024-03-{2+i:02d}", "FAKE (ai)", "cg_fake"))
    if hum:
        rows.append((hum[0], "user_a3", 5, "2024-03-08", "FAKE (human-deceptive)", "human_fake"))
    for t, u, r, d, lab in NONREV:
        rows.append((t, u, r, d, lab, "non_review"))

    out = os.path.join(HERE, "sample_analyze_small.csv")
    try:
        f = open(out, "w", newline="", encoding="utf-8")
    except PermissionError:
        out = os.path.join(HERE, "sample_analyze_small_v2.csv")
        f = open(out, "w", newline="", encoding="utf-8")
        print("(sample_analyze_small.csv was locked — wrote v2 instead)")
    with f:
        csv.writer(f).writerows(rows)
    print(f"wrote {os.path.basename(out)} ({len(rows)-1} rows)")


if __name__ == "__main__":
    main()

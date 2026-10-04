# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
shrink_sentiment_rf.py — retrain ONLY the sentiment Random Forest with bounded tree
size, replacing the bloated data/sentiment_rf.joblib (~2.19 GB).

WHY it was huge: the original RF used unbounded depth (no max_depth / min_samples_leaf),
so each of its 100 trees grew until every leaf was pure on 500k samples -> hundreds of
millions of nodes -> 2.19 GB (and the same in RAM, since the models are saved
uncompressed). Deep unbounded RF on TF-IDF mostly memorizes the training set.

THE FIX: cap max_depth + min_samples_leaf. This shrinks the model ~10-40x with
negligible accuracy change. The script prints test accuracy + the new size so you can
verify; tune the two constants if you want it smaller (raise MIN_SAMPLES_LEAF / lower
MAX_DEPTH) or more accurate (the reverse). The old file is backed up to .bak.

Run:  python shrink_sentiment_rf.py
"""

import os
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# reuse the EXACT TF-IDF config + data loader the full trainer uses (importing is safe —
# train_merged_sentiment_models has an `if __name__ == "__main__"` guard)
from train_merged_sentiment_models import load_data, make_pipeline, print_metrics, SAMPLE_SIZE

# ── tune these two to trade size vs accuracy ──
MAX_DEPTH        = 40
MIN_SAMPLES_LEAF = 5

OUT = os.path.join(os.path.dirname(__file__), "data", "sentiment_rf.joblib")


def _fmt(b):
    return f"{b/1e9:.2f} GB" if b >= 1e9 else f"{b/1e6:.0f} MB"


def main():
    old = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    print(f"Current sentiment_rf.joblib: {_fmt(old) if old else 'absent'}")

    print(f"Loading {SAMPLE_SIZE:,} samples (same data + split as the full trainer)...")
    X, y = load_data(SAMPLE_SIZE)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = make_pipeline(
        RandomForestClassifier(n_estimators=100, max_depth=MAX_DEPTH,
                               min_samples_leaf=MIN_SAMPLES_LEAF, class_weight="balanced",
                               random_state=42, n_jobs=-1),
        max_features=5_000,
    )
    print(f"Training RF (max_depth={MAX_DEPTH}, min_samples_leaf={MIN_SAMPLES_LEAF})...")
    t = time.time()
    pipe.fit(X_tr, y_tr)
    print(f"  fit in {time.time() - t:.0f}s")

    print("\nTest-set metrics (compare to the original RF's numbers in your training logs):")
    print_metrics(y_te, pipe.predict(X_te))

    # back up the old model, then save the new one UNCOMPRESSED so the file size ≈ the
    # RAM footprint (honest comparison; the server loads it as-is)
    if old:
        bak = OUT + ".bak"
        if os.path.exists(bak):
            os.remove(OUT)
        else:
            os.replace(OUT, bak)
            print(f"\nBacked up old model -> {bak}")
    joblib.dump(pipe, OUT)

    new = os.path.getsize(OUT)
    ratio = f"  ({old / new:.0f}x smaller)" if old and new else ""
    print(f"New sentiment_rf.joblib: {_fmt(new)}{ratio}")
    print("\nDone. Restart the server to load the smaller model. Once you've confirmed")
    print("accuracy looks fine, delete data/sentiment_rf.joblib.bak to reclaim the space.")


if __name__ == "__main__":
    main()

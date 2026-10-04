# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
make_colab_dl_bundle_all.py — bundle the 8 DL models for the 4-detector thesis comparison
(GPU re-scoring on the CORRECTED frontend), + the corrected frontend CSV. One clean run, no
reuse of the pre-chi-fix dl_predictions.csv.

Detectors -> their DL members (sklearn members are run locally in the merge):
  baseline    : yelp_fake_lstm_onnx, yelp_fake_distilbert_onnx
  focal       : yelp_fake_lstm_focal_onnx, yelp_fake_distilbert_focal_onnx
  contrastive : yelp_fake_lstm_contrastive_onnx, yelp_fake_distilbert_contrastive_onnx
  modelb      : yelp_multimodal_lstm_full_onnx, yelp_multimodal_distilbert_full_onnx

Output: colab_dl_bundle_all.zip   RUN: python make_colab_dl_bundle_all.py
"""
import os, zipfile

HERE = os.path.dirname(__file__); DATA = os.path.join(HERE, "data")
DIRS = ["yelp_fake_lstm_onnx", "yelp_fake_distilbert_onnx",
        "yelp_fake_lstm_focal_onnx", "yelp_fake_distilbert_focal_onnx",
        "yelp_fake_lstm_contrastive_onnx", "yelp_fake_distilbert_contrastive_onnx",
        "yelp_multimodal_lstm_full_onnx", "yelp_multimodal_distilbert_full_onnx"]
CSV = "yelp_frontend_multimodal.csv"
OUT = os.path.join(HERE, "colab_dl_bundle_all.zip")


def main():
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(os.path.join(DATA, CSV), arcname=CSV)
        for d in DIRS:
            base = os.path.join(DATA, d)
            if not os.path.isdir(base):
                print(f"  MISSING: {d}"); continue
            for root, _, files in os.walk(base):
                for fn in files:
                    full = os.path.join(root, fn)
                    zf.write(full, arcname=os.path.relpath(full, DATA))
            print(f"  added {d}")
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")
    print("Upload to Colab/Drive, run compare_dl_all_colab.ipynb (GPU) -> dl_predictions_all.csv")


if __name__ == "__main__":
    main()

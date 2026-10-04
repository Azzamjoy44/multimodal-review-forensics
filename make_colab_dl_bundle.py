"""
make_colab_dl_bundle.py — bundle everything the Colab DL-comparison notebook needs into ONE
zip you upload to Colab/Drive: the 6 DL model dirs (3 LSTM + 3 DistilBERT, across baseline /
old Model B / new pure-full Model B) + the corrected frontend serving CSV.

Output: colab_dl_bundle.zip  (~0.8 GB — mostly the 3 DistilBERT .onnx files)
RUN: python make_colab_dl_bundle.py
"""
import os
import zipfile

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
DIRS = [
    "yelp_fake_lstm_onnx", "yelp_fake_distilbert_onnx",                       # baseline (text-only)
    "yelp_multimodal_lstm_onnx", "yelp_multimodal_distilbert_onnx",           # OLD Model B (AI-aug)
    "yelp_multimodal_lstm_full_onnx", "yelp_multimodal_distilbert_full_onnx", # NEW pure-full Model B
]
CSV = "yelp_frontend_multimodal.csv"
OUT = os.path.join(HERE, "colab_dl_bundle.zip")


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
                    arc = os.path.relpath(full, DATA)          # keep dir structure under data/
                    zf.write(full, arcname=arc)
            print(f"  added {d}")
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB)")
    print("Upload colab_dl_bundle.zip to your Colab session (or Drive), then run compare_dl_colab.ipynb.")


if __name__ == "__main__":
    main()

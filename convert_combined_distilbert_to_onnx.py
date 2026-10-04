"""
convert_combined_distilbert_to_onnx.py — export the UNIFIED (CG + human deception) DistilBERT
from HF safetensors (data/combined_fake_distilbert/) to ONNX (data/combined_fake_distilbert_onnx/)
so the live server can serve it via onnxruntime exactly like Model A's DistilBERT (no torch /
no TensorFlow at runtime). Input: input_ids + attention_mask (int64). Output: logits[batch, 2].

RUN (once, locally — needs torch + transformers):  python convert_combined_distilbert_to_onnx.py
"""

import os
os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("USE_TORCH", "1")
import json
import shutil

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(__file__)
SRC  = os.path.join(HERE, "data", "combined_fake_distilbert")
OUT  = os.path.join(HERE, "data", "combined_fake_distilbert_onnx")
MAX_LEN = 256


class _Wrap(torch.nn.Module):
    """Return the raw logits tensor so torch.onnx.export traces a plain tensor output."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


def main():
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForSequenceClassification.from_pretrained(SRC).eval()
    wrap = _Wrap(model).eval()

    enc = tok("placeholder review text for tracing", truncation=True, padding="max_length",
              max_length=MAX_LEN, return_tensors="pt")
    args = (enc["input_ids"], enc["attention_mask"])

    onnx_path = os.path.join(OUT, "model.onnx")
    torch.onnx.export(
        wrap, args, onnx_path,
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=14, do_constant_folding=True, dynamo=False,
    )
    # tokenizer files alongside the onnx so AutoTokenizer.from_pretrained(OUT) works at serving
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"):
        src = os.path.join(SRC, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, f))
    json.dump({"threshold": 0.5}, open(os.path.join(OUT, "threshold.json"), "w"), indent=2)
    print(f"wrote {onnx_path}")

    # sanity check: run the ONNX model and compare to torch logits
    import numpy as np, onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {"input_ids": enc["input_ids"].numpy().astype("int64"),
            "attention_mask": enc["attention_mask"].numpy().astype("int64")}
    onnx_logits = sess.run(None, feed)[0]
    with torch.no_grad():
        torch_logits = wrap(*args).numpy()
    print("torch logits:", torch_logits.round(4), "| onnx logits:", onnx_logits.round(4),
          "| max abs diff:", float(np.abs(torch_logits - onnx_logits).max()))


if __name__ == "__main__":
    main()

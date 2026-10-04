# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
train_human_fake_distilbert.py — DistilBERT detector of HUMAN-WRITTEN deceptive reviews,
fine-tuned on the Ott + Li 2014 gold corpus (data/human_fake/human_fake_reviews.csv).

The DL counterpart to train_human_fake_sklearn.py. Small dataset (~2.3k train) so we
fine-tune for a few epochs with EARLY STOPPING on val macro-F1 (restore best) + weighted
cross-entropy for the mild class imbalance. Runs on CPU (the corpus is tiny).

Output: data/human_fake/human_fake_distilbert/ (HF model + tokenizer)
RUN:    python train_human_fake_distilbert.py
"""

import os
# Force the torch backend and stop transformers from importing TensorFlow (its protobuf
# gencode is incompatible with the installed runtime -> import crash). We only want torch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, precision_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "human_fake", "human_fake_reviews.csv")
OUT  = os.path.join(HERE, "data", "human_fake", "human_fake_distilbert")
MODEL_NAME, MAX_LEN, BATCH, LR, MAX_EPOCHS, PATIENCE = "distilbert-base-uncased", 256, 16, 2e-5, 8, 2


def encode(tok, texts):
    e = tok(list(texts), truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


def main():
    torch.manual_seed(0)
    df = pd.read_csv(DATA); df["text"] = df["text"].astype(str)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"train {len(tr):,} / val {len(va):,} / test {len(te):,}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)

    def loader(d, shuffle):
        ids, am = encode(tok, d.text); y = torch.tensor(d.label.values)
        return DataLoader(TensorDataset(ids, am, y), batch_size=BATCH, shuffle=shuffle)
    tr_dl, va_dl, te_dl = loader(tr, True), loader(va, False), loader(te, False)

    n0, n1 = (tr.label == 0).sum(), (tr.label == 1).sum()
    w = torch.tensor([len(tr)/(2*n0), len(tr)/(2*n1)], dtype=torch.float, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    @torch.no_grad()
    def evaluate(dl):
        model.eval(); yt, yp = [], []
        for ids, am, y in dl:
            logit = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits
            yp += logit.argmax(1).cpu().tolist(); yt += y.tolist()
        return (f1_score(yt, yp, average="macro", zero_division=0),
                recall_score(yt, yp, pos_label=1, zero_division=0),
                precision_score(yt, yp, pos_label=1, zero_division=0))

    best, best_state, bad = -1, None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for ids, am, y in tr_dl:
            opt.zero_grad()
            logit = model(input_ids=ids.to(dev), attention_mask=am.to(dev)).logits
            F.cross_entropy(logit, y.to(dev), weight=w).backward()
            opt.step()
        vmf1 = evaluate(va_dl)[0]
        print(f"epoch {ep}: val macro-F1 {vmf1:.1%}", flush=True)
        if vmf1 > best:
            best, best_state, bad = vmf1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= PATIENCE:
            print(f"early stop at epoch {ep}", flush=True); break
    model.load_state_dict(best_state)

    mf1, fr, fp = evaluate(te_dl)
    print("=" * 56)
    print(f"DistilBERT human-deception — test: macro-F1 {mf1:.1%} | fake recall {fr:.1%} | fake precision {fp:.1%}")
    print("=" * 56, flush=True)
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

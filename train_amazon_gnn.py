"""
train_amazon_gnn.py — GNN fraud/ring detector on the Amazon (Musical Instruments)
benchmark (CARE-GNN / GADBench), the second GNN dataset alongside Yelp.

Graph: 11,944 reviewer nodes, 25 handcrafted features, 3 relations
  net_upu — reviewers who reviewed the SAME product (the co-review / ring relation)
  net_usu — reviewers with the same star rating in the same week
  net_uvu — reviewers with top-5% review-text TF-IDF similarity
Label: paid fake reviewer (1) vs benign (0). NOTE the CARE-GNN quirk — nodes 0..3304 are
UNLABELED placeholders; the real labeled set is nodes 3305.. (9.5% fraud). We split ONLY the
labeled nodes 80/10/10 (semi-supervised: all nodes still pass messages).

Model: 2-layer multi-relation mean-aggregation GraphSAGE (pure torch — the graph is tiny,
no torch_geometric / GPU needed). Methodologically the same relational-GraphSAGE family as
the Yelp GNN. Trains on CPU in seconds.

Outputs: data/amazon_gnn/amazon_gnn_preds.npz (prob_fraud for ALL nodes + label + split)
RUN:  python train_amazon_gnn.py
"""

import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(__file__)
MAT  = os.path.join(HERE, "data", "amazon_gnn", "Amazon.mat")
OUT  = os.path.join(HERE, "data", "amazon_gnn", "amazon_gnn_preds.npz")
RELS = ("upu", "usu", "uvu")
N_UNLABELED = 3305     # CARE-GNN: first 3305 Amazon nodes are unlabeled placeholders


def norm_adj(A):
    """Row-normalized sparse adjacency with self-loops -> torch sparse (mean aggregation)."""
    A = A.tocoo().astype(np.float32)
    A = (A + sp.eye(A.shape[0], dtype=np.float32)).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    A = sp.diags(1.0 / np.maximum(deg, 1.0)) @ A
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()


class RGNN(torch.nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.s1 = torch.nn.Linear(in_dim, hid)
        self.r1 = torch.nn.ModuleDict({r: torch.nn.Linear(in_dim, hid) for r in RELS})
        self.s2 = torch.nn.Linear(hid, hid)
        self.r2 = torch.nn.ModuleDict({r: torch.nn.Linear(hid, hid) for r in RELS})
        self.out = torch.nn.Linear(hid, 2)
        self.drop = torch.nn.Dropout(0.3)

    def forward(self, x, adjs):
        h = self.s1(x) + sum(self.r1[r](torch.sparse.mm(adjs[r], x)) for r in RELS)
        h = self.drop(F.relu(h))
        h = self.s2(h) + sum(self.r2[r](torch.sparse.mm(adjs[r], h)) for r in RELS)
        h = self.drop(F.relu(h))
        return self.out(h)


def main():
    m = sio.loadmat(MAT)
    X = np.asarray(m["features"].todense(), dtype=np.float32)
    y = m["label"].ravel().astype(np.int64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Xt = torch.tensor(X)
    adjs = {r: norm_adj(m[f"net_{r}"]) for r in RELS}
    print(f"nodes {len(y):,} | feat {X.shape[1]} | fraud(all) {int((y==1).sum()):,} "
          f"({(y==1).mean():.1%})  [incl. {N_UNLABELED} unlabeled placeholders]")

    lab_idx = np.arange(N_UNLABELED, len(y))
    print(f"labeled nodes {len(lab_idx):,} | fraud(labeled) {(y[lab_idx]==1).mean():.1%}")
    tr, tmp = train_test_split(lab_idx, test_size=0.20, stratify=y[lab_idx], random_state=42)
    va, te  = train_test_split(tmp,    test_size=0.50, stratify=y[tmp],     random_state=42)
    yt = torch.tensor(y)
    n0, n1 = (y[tr] == 0).sum(), (y[tr] == 1).sum()
    w = torch.tensor([len(tr)/(2*n0), len(tr)/(2*n1)], dtype=torch.float)
    print(f"split: train {len(tr):,} / val {len(va):,} / test {len(te):,}  (class weights {w.tolist()})")

    torch.manual_seed(0)
    model = RGNN(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=5e-4)

    def evaluate(mask):
        model.eval()
        with torch.no_grad():
            logit = model(Xt, adjs)
            prob = torch.softmax(logit, 1)[:, 1].numpy()
            pred = logit.argmax(1).numpy()
        return (f1_score(y[mask], pred[mask], average="macro", zero_division=0),
                recall_score(y[mask], pred[mask], pos_label=1, zero_division=0),
                precision_score(y[mask], pred[mask], pos_label=1, zero_division=0),
                roc_auc_score(y[mask], prob[mask]), prob)

    best, best_state, bad = -1, None, 0
    for ep in range(1, 301):
        model.train(); opt.zero_grad()
        out = model(Xt, adjs)
        loss = F.cross_entropy(out[tr], yt[tr], weight=w)
        loss.backward(); opt.step()
        vmf1 = evaluate(va)[0]
        if vmf1 > best:
            best, best_state, bad = vmf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 25:
            break
    model.load_state_dict(best_state)

    mf1, fr, fp, auc, prob = evaluate(te)
    print("=" * 64)
    print(f"AMAZON GNN — test (n={len(te):,}): macro-F1 {mf1:.1%} | fraud recall {fr:.1%} "
          f"| fraud precision {fp:.1%} | AUC {auc:.3f}")
    print("=" * 64)

    split = np.zeros(len(y), dtype=np.int64)   # 0 unlabeled/none, 1 train, 2 val, 3 test
    split[tr] = 1; split[va] = 2; split[te] = 3
    np.savez_compressed(OUT, prob_fraud=prob, label=y, split=split)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

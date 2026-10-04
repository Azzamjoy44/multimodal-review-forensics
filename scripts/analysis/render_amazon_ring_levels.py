# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
render_amazon_ring_levels.py — coordinated-ring graphs for the Amazon (Musical Instruments)
GNN, at several precision/coverage operating points, for the frontend's precision selector.

The GNN flags fake reviewers (top ~9.5% by P(fraud), the calibrated rate). Rings are
extracted from the net_upu relation (reviewers who reviewed the same products): we induce
the flagged-reviewer subgraph and take its k-core — k = how many flagged co-reviewers each
member must connect to (the Amazon analog of Yelp's "shared businesses" strictness). Higher
k = tighter, higher-precision rings. Validation uses the real labels (only labeled nodes,
idx >= 3305 — the first 3305 Amazon nodes are unlabeled placeholders).

Outputs static/gnn/levels/amazon_rings_k{K}.png + upserts the "Amazon (Musical Instruments)"
entry into the shared manifest.  RUN:  python render_amazon_ring_levels.py
"""

import os
import math
from collections import defaultdict

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import networkx as nx

from gnn_manifest import upsert_dataset

HERE   = os.path.dirname(__file__)
MAT    = os.path.join(HERE, "data", "amazon_gnn", "Amazon.mat")
PREDS  = os.path.join(HERE, "data", "amazon_gnn", "amazon_gnn_preds.npz")
OUTDIR = os.path.join(HERE, "static", "gnn", "levels")
N_UNLABELED, MIN_REVIEWERS, MAX_NODES = 3305, 4, 60
LEVELS = [(1, "Broad coverage"), (2, "Balanced"), (3, "Strict")]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    p = np.load(PREDS)
    prob, label = p["prob_fraud"], p["label"]
    true_rate = float((label[N_UNLABELED:] == 1).mean())          # labeled fraud rate (~9.5%)
    k_flag = int(round(true_rate * len(prob)))
    flagged = prob >= np.partition(prob, -k_flag)[-k_flag]
    flagged_set = set(np.nonzero(flagged)[0].tolist())
    print(f"flagging top {true_rate:.1%} by P(fraud) -> {int(flagged.sum()):,} reviewers")

    # net_upu subgraph among flagged reviewers
    A = sio.loadmat(MAT)["net_upu"].tocoo()
    G = nx.Graph()
    for i, j in zip(A.row.tolist(), A.col.tolist()):
        if i < j and i in flagged_set and j in flagged_set:
            G.add_edge(i, j)
    print(f"flagged-reviewer co-review graph: {G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges")

    is_fraud   = lambda u: label[u] == 1
    is_labeled = lambda u: u >= N_UNLABELED

    def member_prec(members):
        lab = [u for u in members if is_labeled(u)]
        return (sum(1 for u in lab if is_fraud(u)) / len(lab)) if lab else float("nan")

    def render(rings, k, header=""):
        n = max(1, len(rings)); ncols = min(4, n); nrows = max(1, math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.4), squeeze=False)
        axes = axes.ravel()
        for ax, comp in zip(axes, rings):
            sub = G.subgraph(comp)
            if sub.number_of_nodes() > MAX_NODES:                  # keep the densest core readable
                top = sorted(sub.degree, key=lambda x: -x[1])[:MAX_NODES]
                sub = sub.subgraph([u for u, _ in top])
            pos = nx.spring_layout(sub, k=0.5, iterations=80, seed=3)
            cols = ["#ef4444" if is_fraud(u) else ("#9ca3af" if is_labeled(u) else "#dbe4f0")
                    for u in sub.nodes]
            nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.20, width=0.5)
            nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=cols, node_size=110,
                                   edgecolors="#1f2937", linewidths=0.5)
            ax.set_title(f"{len(comp)} reviewers · {member_prec(comp):.0%} fraud", fontsize=10, weight="bold")
            ax.axis("off")
        for ax in axes[len(rings):]:
            ax.axis("off")
        fig.legend(handles=[mp.Patch(color="#ef4444", label="reviewer — confirmed fake (correct catch)"),
                            mp.Patch(color="#9ca3af", label="reviewer — benign (false positive)"),
                            mp.Patch(color="#dbe4f0", label="reviewer — unlabeled (status unknown)")],
                   loc="lower center", ncol=3, fontsize=11, frameon=False)
        if header:
            fig.suptitle(header, fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout(rect=[0, 0.04, 1, 0.94])
        fn = f"amazon_rings_k{k}.png"
        plt.savefig(os.path.join(OUTDIR, fn), dpi=130, bbox_inches="tight"); plt.close(fig)
        return f"/static/gnn/levels/{fn}"

    levels = []
    for k, lbl in LEVELS:
        core = nx.k_core(G, k) if G.number_of_nodes() else nx.Graph()
        rings = sorted((c for c in nx.connected_components(core) if len(c) >= MIN_REVIEWERS),
                       key=len, reverse=True)
        members = {u for r in rings for u in r}
        prec = member_prec(members)
        rrp = (sum(1 for u in members if is_labeled(u) and is_fraud(u)) /
               max(sum(1 for u in members if is_labeled(u)), 1))
        if not rings:
            print(f"k={k}: 0 rings (skipped)"); continue
        url = render(rings, k,
                     header=f"Amazon Musical Instruments — {lbl.title()}   ·   {len(rings)} rings · {prec:.0%} precision")
        levels.append({
            "min_shared": k, "label": lbl, "n_rings": len(rings), "n_reviewers": len(members),
            "review_prec": round(prec, 3), "reviewer_prec": round(rrp, 3),
            "enrichment": round(prec / true_rate, 1), "url": url,
            "caption": (f"{lbl}: {len(rings)} rings / {len(members)} reviewers · {prec:.0%} of "
                        f"labeled members are confirmed fake ({prec/true_rate:.1f}× the {true_rate:.0%} "
                        f"base rate). Reviewers linked by shared products; ring = a {k}-core of "
                        f"flagged reviewers. (net_upu collapses product counts, so strictness is the "
                        f"k-core depth, not shared-product count.)"),
        })
        print(f"k={k}: {len(rings)} rings / {len(members)} reviewers / {prec:.0%} fraud-precision -> {url}")

    upsert_dataset("Amazon (Musical Instruments)", levels)
    print(f"upserted Amazon into the manifest ({len(levels)} levels)")


if __name__ == "__main__":
    main()

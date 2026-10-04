"""
visualize_gnn_rings.py — draw the GNN-detected coordinated rings (corrected graph + plain
co-edges). Each panel is one ring: nodes = reviewers (circles) + the businesses they jointly
target (blue squares), edges = the GNN-flagged reviews.

Reviewer colour encodes the GROUND-TRUTH validation (the GNN never saw it at ring time):
  * red  = truly fake reviewer (>=1 confirmed fake review) — a correct catch
  * grey = genuine reviewer    (no fake reviews)           — a false positive
so each panel visually shows the ring's precision (how red it is).

Reads data/yelp_graph/yelp_rings_corrected.csv (reviewer indices) +
yelp_gnn_temporal_preds.npz (review_user / review_biz / prob_co / label).

RUN:  python visualize_gnn_rings.py   ->  yelp_rings_gnn_corrected_viz.png
"""

import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import networkx as nx

GDIR     = os.path.join(os.path.dirname(__file__), "data", "yelp_graph")
PREDS    = os.path.join(GDIR, "yelp_gnn_temporal_preds.npz")
RINGS    = os.path.join(GDIR, "yelp_rings_corrected.csv")
OUT      = os.path.join(os.path.dirname(__file__), "static", "gnn", "yelp_rings_gnn_corrected_viz.png")
MAX_BIZ  = 30    # draw at most this many (most-shared) businesses per ring, for readability
N_PANELS = 6


def main():
    p = np.load(PREDS)
    ru, rb, lab, prob = p["review_user"], p["review_biz"], p["label"], p["prob_co"]
    true_rate = float((lab == 1).mean())
    k = int(round(true_rate * len(prob)))
    flagged = prob >= np.partition(prob, -k)[-k]

    # flagged businesses per reviewer + true-fake status per reviewer
    biz_by_user = defaultdict(set)
    for u, b in zip(ru[flagged].tolist(), rb[flagged].tolist()):
        biz_by_user[u].add(b)
    user_fake = defaultdict(lambda: [0, 0])
    for i in range(len(ru)):
        if lab[i] in (0, 1):
            user_fake[int(ru[i])][int(lab[i])] += 1
    is_fake = lambda u: user_fake[u][1] > 0

    rings = []
    for row in csv.DictReader(open(RINGS, encoding="utf-8")):
        rings.append({"id": int(row["ring_id"]), "n": int(row["n_reviewers"]),
                      "fr": float(row["member_true_fake_rate"]),
                      "members": [int(x) for x in row["reviewer_idxs"].split("|") if x != ""]})
    # render EVERY detected ring (size-sorted in the CSV), in a dynamic grid
    import math
    ncols = 4
    nrows = max(1, math.ceil(len(rings) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, rings):
        members = r["members"]
        bizcount = defaultdict(int)
        for u in members:
            for b in biz_by_user[u]:
                bizcount[b] += 1
        shared = sorted((b for b, c in bizcount.items() if c >= 2), key=lambda b: -bizcount[b])[:MAX_BIZ]
        sset = set(shared)

        G = nx.Graph()
        G.add_nodes_from((("r", u) for u in members))
        G.add_nodes_from((("b", b) for b in shared))
        for u in members:
            for b in biz_by_user[u] & sset:
                G.add_edge(("r", u), ("b", b))
        pos = nx.spring_layout(G, k=0.55, iterations=90, seed=3)
        rnodes = [n for n in G if n[0] == "r"]
        rcolors = ["#ef4444" if is_fake(n[1]) else "#9ca3af" for n in rnodes]
        bnodes = [n for n in G if n[0] == "b"]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.22, width=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=bnodes, ax=ax, node_color="#3b82f6",
                               node_shape="s", node_size=55)
        nx.draw_networkx_nodes(G, pos, nodelist=rnodes, ax=ax, node_color=rcolors,
                               node_size=120, edgecolors="#1f2937", linewidths=0.6)
        ax.set_title(f"Ring {r['id']}:  {r['n']} reviewers → {len(shared)} biz · {r['fr']:.0%} fake",
                     fontsize=10, weight="bold")
        ax.axis("off")
    for ax in axes[len(rings):]:
        ax.axis("off")

    fig.legend(handles=[mp.Patch(color="#ef4444", label="reviewer — truly fake (correct catch)"),
                        mp.Patch(color="#9ca3af", label="reviewer — genuine (false positive)"),
                        mp.Patch(color="#3b82f6", label="business (shared target)")],
               loc="lower center", ncol=3, fontsize=12, frameon=False)
    fig.suptitle(f"GNN-detected coordinated fake-review rings — ALL {len(rings)} rings "
                 f"(corrected graph + co-edges, label-free)\n"
                 "reviewers converging on the same businesses; colour = ground-truth validation",
                 fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"saved {OUT}  ({len(rings)} rings drawn)")


if __name__ == "__main__":
    main()

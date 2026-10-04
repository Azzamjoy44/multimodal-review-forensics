"""
render_gnn_ring_levels.py — render the GNN's coordinated-ring graphs at several
PRECISION/COVERAGE operating points, for the frontend's precision selector.

The knob is link-strictness MIN_SHARED (how many shared flagged businesses link two
reviewers): looser = more rings but noisier, stricter = fewer but higher precision.
For each level it extracts the rings from the co-edge GNN's predictions (prob_co),
renders every ring (colour = ground-truth check), and records the precision/coverage
stats so the UI can show them.

Outputs:
  static/gnn/levels/rings_ms{N}.png   — all rings at strictness N
  static/gnn/levels/manifest.json     — [{min_shared, label, n_rings, n_reviewers,
                                          review_prec, reviewer_prec, enrichment, url, caption}]

RUN:  python render_gnn_ring_levels.py
"""

import os
import json
import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import networkx as nx

HERE   = os.path.dirname(__file__)
PREDS  = os.path.join(HERE, "data", "yelp_graph", "yelp_gnn_temporal_preds.npz")
OUTDIR = os.path.join(HERE, "static", "gnn", "levels")
CAP, MIN_REVIEWERS, MAX_BIZ = 40, 4, 30
LEVELS = [(2, "Broad coverage"), (3, "Balanced"), (4, "Strict")]


def rings_from(is_fake, ru, rb, min_shared):
    biz = defaultdict(set)
    for i in np.nonzero(is_fake)[0].tolist():
        biz[int(rb[i])].add(int(ru[i]))
    pair = defaultdict(int)
    for rs in biz.values():
        if 2 <= len(rs) <= CAP:
            for a, c in combinations(sorted(rs), 2):
                pair[(a, c)] += 1
    parent = {}
    def find(x):
        parent.setdefault(x, x); root = x
        while parent[root] != root: root = parent[root]
        while parent[x] != root: parent[x], x = root, parent[x]
        return root
    for (a, c), cnt in pair.items():
        if cnt >= min_shared:
            ra, rc = find(a), find(c)
            if ra != rc: parent[ra] = rc
    comp = defaultdict(list)
    for u in list(parent):
        comp[find(u)].append(u)
    return sorted((m for m in comp.values() if len(m) >= MIN_REVIEWERS), key=len, reverse=True)


def render(rings, ms, biz_by_user, user_fake, header=""):
    is_fake = lambda u: user_fake[u][1] > 0
    n = max(1, len(rings))
    ncols = min(4, n)
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.4), squeeze=False)
    axes = axes.ravel()
    for ax, members in zip(axes, rings):
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
        rn = [x for x in G if x[0] == "r"]
        rc = ["#ef4444" if is_fake(x[1]) else "#9ca3af" for x in rn]
        bn = [x for x in G if x[0] == "b"]
        g0 = sum(user_fake[u][0] for u in members); f1 = sum(user_fake[u][1] for u in members)
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.22, width=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=bn, ax=ax, node_color="#3b82f6", node_shape="s", node_size=55)
        nx.draw_networkx_nodes(G, pos, nodelist=rn, ax=ax, node_color=rc, node_size=120,
                               edgecolors="#1f2937", linewidths=0.6)
        ax.set_title(f"{len(members)} reviewers → {len(shared)} biz · {f1/max(g0+f1,1):.0%} fake",
                     fontsize=10, weight="bold")
        ax.axis("off")
    for ax in axes[len(rings):]:
        ax.axis("off")
    fig.legend(handles=[mp.Patch(color="#ef4444", label="reviewer — truly fake (correct catch)"),
                        mp.Patch(color="#9ca3af", label="reviewer — genuine (false positive)"),
                        mp.Patch(color="#3b82f6", label="business (shared target)")],
               loc="lower center", ncol=3, fontsize=11, frameon=False)
    if header:
        fig.suptitle(header, fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    fn = f"rings_ms{ms}.png"
    plt.savefig(os.path.join(OUTDIR, fn), dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"/static/gnn/levels/{fn}"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    p = np.load(PREDS)
    ru, rb, lab = p["review_user"], p["review_biz"], p["label"]
    # on the deduped graph CO-temporal is the best ring config (60% vs CO-plain 49%); fall back to prob_co
    prob = p["prob_co_temporal"] if "prob_co_temporal" in p.files else p["prob_co"]
    true_rate = float((lab == 1).mean())
    k = int(round(true_rate * len(prob)))
    flagged = prob >= np.partition(prob, -k)[-k]

    biz_by_user = defaultdict(set)
    for u, b in zip(ru[flagged].tolist(), rb[flagged].tolist()):
        biz_by_user[u].add(b)
    user_fake = defaultdict(lambda: [0, 0])
    for i in range(len(ru)):
        if lab[i] in (0, 1):
            user_fake[int(ru[i])][int(lab[i])] += 1

    manifest = []
    for ms, label in LEVELS:
        rings = rings_from(flagged, ru, rb, ms)
        if not rings:                               # skip empty operating points (no high-prec tier on clean graph)
            print(f"ms={ms}: 0 rings — skipped"); continue
        members = {u for r in rings for u in r}
        g = sum(user_fake[u][0] for u in members); f = sum(user_fake[u][1] for u in members)
        rf = sum(1 for u in members if user_fake[u][1] > 0)
        rvp = f / max(g + f, 1)
        rrp = rf / max(len(members), 1)
        url = render(rings, ms, biz_by_user, user_fake,
                     header=f"Yelp — {label.title()}   ·   {len(rings)} rings · {rvp:.0%} precision")
        manifest.append({
            "min_shared": ms, "label": label,
            "n_rings": len(rings), "n_reviewers": len(members),
            "review_prec": round(rvp, 3), "reviewer_prec": round(rrp, 3),
            "enrichment": round(rvp / true_rate, 1), "url": url,
            "caption": (f"{label}: {len(rings)} rings / {len(members)} reviewers · "
                        f"{rvp:.0%} review-precision, {rrp:.0%} reviewer-precision "
                        f"({rvp/true_rate:.1f}× the {true_rate:.0%} base rate). "
                        f"Link reviewers sharing ≥{ms} flagged businesses."),
        })
        print(f"ms={ms}: {len(rings)} rings / {len(members)} reviewers / {rvp:.0%} prec -> {url}")

    from gnn_manifest import upsert_dataset
    upsert_dataset("Yelp", manifest)
    print("upserted Yelp into the shared manifest")


if __name__ == "__main__":
    main()

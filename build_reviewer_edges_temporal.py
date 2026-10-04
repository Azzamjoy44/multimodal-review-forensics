"""
build_reviewer_edges_temporal.py — TEMPORAL-BURST reviewer<->reviewer co-edges.

Sharpens the plain co-edges (build_reviewer_edges.py) with the key ring signal: coordinated
crews hit the same targets in BURSTS, genuine co-reviewers are spread over years. So we link
two reviewers iff they share >= K businesses AND on >= 1 of those shared businesses their
reviews fall within W days of each other. A strict subset of the plain edges -> higher-quality
coordination signal for the GNN.

Reads data/yelp_graph/graph_full.npz (review_user, review_biz, review_day) ->
data/yelp_graph/reviewer_edges_temporal.npz (edge_index, 2 x E, bidirectional).

RUN:  python build_reviewer_edges_temporal.py
"""

import os
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np

GDIR = os.path.join(os.path.dirname(__file__), "data", "yelp_graph")
CAP = 150  # ignore businesses with > CAP reviewers (bounds the pair count)
K   = 2    # share >= K businesses
W   = 30   # ... and co-reviewed >= 1 of them within W days (the burst window)


def main():
    g = np.load(os.path.join(GDIR, "graph_full.npz"))
    ru, rb, rd = g["review_user"], g["review_biz"], g["review_day"]
    # business -> list of (reviewer_idx, day); day -1 means unparseable
    biz = defaultdict(list)
    for u, b, day in zip(ru.tolist(), rb.tolist(), rd.tolist()):
        biz[b].append((u, day))

    shared = defaultdict(int)   # (a,c) -> # shared businesses
    burst  = defaultdict(int)   # (a,c) -> # shared businesses co-reviewed within W days
    for revs in biz.values():
        if not (2 <= len(revs) <= CAP):
            continue
        revs.sort()
        for (a, da), (c, dc) in combinations(revs, 2):
            if a == c:
                continue
            shared[(a, c)] += 1
            if da >= 0 and dc >= 0 and abs(da - dc) <= W:
                burst[(a, c)] += 1
    edges = [p for p, s in shared.items() if s >= K and burst.get(p, 0) >= 1]
    print(f"candidate pairs: {len(shared):,}  | temporal edges (>= {K} shared & burst<= {W}d): {len(edges):,}")
    print(f"(plain non-temporal edges for reference: see reviewer_edges.npz)")

    if not edges:
        print("no edges — loosen W/K"); return
    ei = np.array(edges, dtype=np.int64).T
    ei = np.concatenate([ei, ei[::-1]], axis=1)
    np.savez_compressed(os.path.join(GDIR, "reviewer_edges_temporal.npz"), edge_index=ei)
    deg = Counter(ei[0].tolist())
    print(f"saved reviewer_edges_temporal.npz | directed edges: {ei.shape[1]:,}")
    print(f"reviewers with >=1 edge: {len(deg):,} | max degree: {max(deg.values()):,} | mean: {sum(deg.values())/len(deg):.1f}")


if __name__ == "__main__":
    main()

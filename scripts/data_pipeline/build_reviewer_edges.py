# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
build_reviewer_edges.py — reviewer<->reviewer co-review edges for the GNN.

The current graph is reviewer->review->business, so two co-reviewing reviewers are 4
hops apart — a 2-layer GNN can't see their coordination. This adds explicit
reviewer<->reviewer edges (two reviewers linked iff they jointly reviewed >= K of the
same businesses) so coordination becomes a direct 1-hop neighbor signal.

Reads data/yelp_graph/graph_full.npz (review_user, review_biz) -> data/yelp_graph/reviewer_edges.npz
(edge_index, 2 x E, bidirectional; reviewer indices aligned with the graph).

RUN:  python build_reviewer_edges.py
"""

import os
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np

GDIR = os.path.join(os.path.dirname(__file__), "data", "yelp_graph")
CAP = 150  # ignore businesses with > CAP reviewers (bounds the pair count); the >=K filter controls noise
K   = 2    # link two reviewers iff they share >= K businesses


def main():
    g = np.load(os.path.join(GDIR, "graph_full.npz"))
    ru, rb = g["review_user"], g["review_biz"]
    biz = defaultdict(set)
    for u, b in zip(ru.tolist(), rb.tolist()):
        biz[b].add(u)
    print(f"businesses: {len(biz):,}  | with 2..{CAP} reviewers: {sum(1 for s in biz.values() if 2<=len(s)<=CAP):,}")

    pair = defaultdict(int)
    for rs in biz.values():
        if 2 <= len(rs) <= CAP:
            for a, c in combinations(sorted(rs), 2):
                pair[(a, c)] += 1
    edges = [(a, c) for (a, c), n in pair.items() if n >= K]
    print(f"candidate pairs: {len(pair):,}  | reviewer<->reviewer edges (>= {K} shared): {len(edges):,}")

    if not edges:
        print("no edges — loosen K or raise CAP"); return
    ei = np.array(edges, dtype=np.int64).T          # 2 x E
    ei = np.concatenate([ei, ei[::-1]], axis=1)      # bidirectional
    np.savez_compressed(os.path.join(GDIR, "reviewer_edges.npz"), edge_index=ei)
    deg = Counter(ei[0].tolist())
    print(f"saved reviewer_edges.npz | directed edges: {ei.shape[1]:,}")
    print(f"reviewers with >=1 co-edge: {len(deg):,} | max degree: {max(deg.values()):,} | "
          f"mean degree: {sum(deg.values())/len(deg):.1f}")


if __name__ == "__main__":
    main()

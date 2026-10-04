"""
sweep_ring_extraction.py — FREE precision tuning of the deployable ring extractor.

No retrain: reuses the co-edge GNN's predictions (prob_co). Sweeps the union-find
link-strictness (MIN_SHARED = how many shared flagged businesses to link two reviewers)
and the minimum ring size, and reports the precision/coverage tradeoff against the full
labels. Stricter linking should drop incidental co-reviewers (the noise dragging the small
rings to ~20-28%) and raise precision.

Reads data/yelp_graph/yelp_gnn_reviewer_edges_preds.npz. Console only.

RUN:  python sweep_ring_extraction.py
"""

import os
from collections import defaultdict
from itertools import combinations

import numpy as np

GDIR  = os.path.join(os.path.dirname(__file__), "data", "yelp_graph")
PREDS = os.path.join(GDIR, "yelp_gnn_reviewer_edges_preds.npz")
CAP   = 40   # ignore mega-target businesses (> CAP flagged reviewers) as noise


def rings(is_fake, ru, rb, min_shared, min_reviewers):
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
            ra, rb_ = find(a), find(c)
            if ra != rb_: parent[ra] = rb_
    comp = defaultdict(list)
    for u in list(parent):
        comp[find(u)].append(u)
    return [m for m in comp.values() if len(m) >= min_reviewers]


def main():
    p = np.load(PREDS)
    ru, rb, lab, prob = p["review_user"], p["review_biz"], p["label"], p["prob_co"]
    true_rate = float((lab == 1).mean())
    k = int(round(true_rate * len(prob)))
    flagged = prob >= np.partition(prob, -k)[-k]

    user_lab = defaultdict(lambda: [0, 0])
    for i in range(len(ru)):
        if lab[i] in (0, 1):
            user_lab[int(ru[i])][int(lab[i])] += 1

    def score(rs):
        members = {u for r in rs for u in r}
        g = sum(user_lab[u][0] for u in members); f = sum(user_lab[u][1] for u in members)
        rev_fake = sum(1 for u in members if user_lab[u][1] > 0)
        return len(rs), len(members), f / max(g + f, 1), rev_fake / max(len(members), 1)

    print(f"flagging top {true_rate:.1%} by prob_co | base rate {true_rate:.1%}")
    print(f"{'min_shared':>10}{'min_size':>9}{'rings':>7}{'reviewers':>11}{'review-prec':>13}{'reviewer-prec':>15}{'enrich':>8}")
    for ms in (2, 3, 4):
        for msize in (4, 6):
            nr, nm, rvp, rrp = score(rings(flagged, ru, rb, ms, msize))
            print(f"{ms:>10}{msize:>9}{nr:>7}{nm:>11}{rvp:>12.0%}{rrp:>14.0%}{rvp/true_rate:>7.1f}x")
    print("\n(current deployable config: min_shared=2, min_size=4 -> 34 rings / 358 / 62%)")


if __name__ == "__main__":
    main()

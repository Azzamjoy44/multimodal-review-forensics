"""
compare_temporal_rings.py — does the temporal-burst co-edge (and the chi fix) improve rings?

Loads data/yelp_graph/yelp_gnn_temporal_preds.npz (prob_base / prob_co / prob_co_temporal,
all trained identically on the CORRECTED graph) and, for each, thresholds to the true ~12.2%
fake rate (matched calibration) and extracts rings at two strictnesses (min_shared 2 = broad,
3 = high-precision, from the free sweep), validating members against the full labels.

RUN (after the notebook's npz is back in data/yelp_graph/):
    python compare_temporal_rings.py
"""

import os
from collections import defaultdict
from itertools import combinations

import numpy as np

GDIR  = os.path.join(os.path.dirname(__file__), "data", "yelp_graph")
PREDS = os.path.join(GDIR, "yelp_gnn_temporal_preds.npz")
CAP   = 40


def rings(is_fake, ru, rb, min_shared, min_reviewers=4):
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
    if not os.path.exists(PREDS):
        print(f"missing {PREDS}\nRun train_yelp_gnn_temporal.ipynb and put the npz here first."); return
    p = np.load(PREDS)
    ru, rb, lab = p["review_user"], p["review_biz"], p["label"]
    true_rate = float((lab == 1).mean())
    k = int(round(true_rate * len(lab)))

    user_lab = defaultdict(lambda: [0, 0])
    for i in range(len(ru)):
        if lab[i] in (0, 1):
            user_lab[int(ru[i])][int(lab[i])] += 1

    def score(rs):
        members = {u for r in rs for u in r}
        g = sum(user_lab[u][0] for u in members); f = sum(user_lab[u][1] for u in members)
        rf = sum(1 for u in members if user_lab[u][1] > 0)
        return len(rs), len(members), f / max(g + f, 1), rf / max(len(members), 1)

    print(f"corrected graph | full labels {len(lab):,} | true fake {true_rate:.1%}")
    for ms in (2, 3):
        print(f"\n--- min_shared = {ms} ({'broad' if ms==2 else 'high-precision'}) ---")
        print(f"{'variant':22}{'rings':>7}{'reviewers':>11}{'review-prec':>13}{'reviewer-prec':>15}{'enrich':>8}")
        for name, key in [('BASE', 'prob_base'), ('CO (plain)', 'prob_co'), ('CO-temporal', 'prob_co_temporal')]:
            if key not in p.files:
                print(f"{name:22}  (missing {key})"); continue
            prob = p[key]
            flagged = prob >= np.partition(prob, -k)[-k]
            nr, nm, rvp, rrp = score(rings(flagged, ru, rb, ms))
            print(f"{name:22}{nr:>7}{nm:>11}{rvp:>12.0%}{rrp:>14.0%}{rvp/true_rate:>7.1f}x")
    print(f"\n(prior buggy-graph CO: min_shared 2 -> 62% | min_shared 3 -> 76%)")


if __name__ == "__main__":
    main()

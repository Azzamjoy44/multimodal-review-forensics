# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
extract_rings_corrected.py — final deployable coordinated-ring extraction.

Uses the RECOMMENDED GNN config: corrected graph (chi reviewer bug fixed) + plain
reviewer<->reviewer co-edges, predictions in yelp_gnn_temporal_preds.npz['prob_co'].

Label-free pipeline: flag the top ~12.2% by P(fake) (the calibrated operating point),
extract rings as connected clusters of reviewers sharing flagged businesses, then consult
the ground-truth labels only to report how real the rings are. Emits BOTH operating points
from the strictness sweep:
  * broad        (MIN_SHARED=2) -> more coverage, ~58% precision
  * high-precision(MIN_SHARED=3) -> fewer, ~78% precision

Reads  data/yelp_graph/yelp_gnn_temporal_preds.npz.
Writes data/yelp_graph/yelp_rings_corrected.csv (broad set; the one we visualize) + console.

RUN:  python extract_rings_corrected.py
"""

import os
import csv
from collections import defaultdict
from itertools import combinations

import numpy as np

GDIR  = os.path.join(os.path.dirname(__file__), "data", "yelp_graph")
PREDS = os.path.join(GDIR, "yelp_gnn_temporal_preds.npz")
OUT   = os.path.join(GDIR, "yelp_rings_corrected.csv")
CAP, MIN_REVIEWERS = 40, 4


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


def main():
    p = np.load(PREDS)
    ru, rb, lab, prob = p["review_user"], p["review_biz"], p["label"], p["prob_co"]
    true_rate = float((lab == 1).mean())
    k = int(round(true_rate * len(prob)))
    thresh = np.partition(prob, -k)[-k]
    flagged = prob >= thresh
    print(f"corrected graph + plain co-edges | flag top {true_rate:.1%} (P>={thresh:.3f}) -> {int(flagged.sum()):,} reviews")

    user_lab = defaultdict(lambda: [0, 0])
    for i in range(len(ru)):
        if lab[i] in (0, 1):
            user_lab[int(ru[i])][int(lab[i])] += 1
    biz_by_user = defaultdict(set)
    for u, b in zip(ru[flagged].tolist(), rb[flagged].tolist()):
        biz_by_user[u].add(b)

    def report(rings, tag):
        members = {u for r in rings for u in r}
        g = sum(user_lab[u][0] for u in members); f = sum(user_lab[u][1] for u in members)
        rf = sum(1 for u in members if user_lab[u][1] > 0)
        rvp = f / max(g + f, 1)
        print(f"  {tag:16} rings {len(rings):>3} | reviewers {len(members):>4} | "
              f"review-prec {rvp:.0%} | reviewer-prec {rf/max(len(members),1):.0%} | enrich {rvp/true_rate:.1f}x")
        return rings

    print("operating points:")
    broad = report(rings_from(flagged, ru, rb, 2), "broad (ms=2)")
    report(rings_from(flagged, ru, rb, 3), "high-prec (ms=3)")

    # save the broad set (the one we visualize) with per-ring detail
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ring_id", "n_reviewers", "n_businesses", "member_true_fake_rate", "reviewer_idxs"])
        for rid, m in enumerate(broad, 1):
            bizs = set().union(*(biz_by_user[u] for u in m)) if m else set()
            g = sum(user_lab[u][0] for u in m); f = sum(user_lab[u][1] for u in m)
            w.writerow([rid, len(m), len(bizs), round(f / max(g + f, 1), 3),
                        "|".join(str(u) for u in sorted(m))])
    print(f"\nwrote {OUT} ({len(broad)} rings, broad set)")
    print(f"\nTop 12 rings (broad):")
    print(f"{'#':>2} {'revwrs':>6} {'bizs':>5} {'true-fake%':>11}")
    for i, m in enumerate(broad[:12], 1):
        bizs = set().union(*(biz_by_user[u] for u in m)) if m else set()
        g = sum(user_lab[u][0] for u in m); f = sum(user_lab[u][1] for u in m)
        print(f"{i:>2} {len(m):>6} {len(bizs):>5} {f/max(g+f,1):>10.0%}")


if __name__ == "__main__":
    main()

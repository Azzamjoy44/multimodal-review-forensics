"""
extract_rings_reviewer_edge.py — the deployable coordinated-ring extraction using the
BEST GNN (reviewer<->reviewer co-edge variant, prob_co).

Label-free pipeline (the actual point of the GNN): the GNN scores every review's P(fake)
from structure + behavior (no labels), we flag the top ~12% (the calibrated operating
point ~ the true Yelp filter rate), and extract rings as connected clusters of reviewers
who jointly hit >=2 of the same flagged businesses. We THEN consult the ground-truth
labels only to report how real the rings are.

Reads  data/yelp_graph/yelp_gnn_reviewer_edges_preds.npz (prob_co + structure + labels).
Writes data/yelp_graph/yelp_rings_reviewer_edge.csv (one row per ring) + console summary.

RUN:  python extract_rings_reviewer_edge.py
"""

import os
import csv
from collections import defaultdict

import numpy as np

from extract_rings_gnn import rings_from   # union-find ring extractor (MIN_SHARED=2, CAP=40, MIN_REVIEWERS=4)

GDIR  = os.path.join(os.path.dirname(__file__), "data", "yelp_graph")
PREDS = os.path.join(GDIR, "yelp_gnn_reviewer_edges_preds.npz")
OUT   = os.path.join(GDIR, "yelp_rings_reviewer_edge.csv")


def main():
    if not os.path.exists(PREDS):
        print(f"missing {PREDS}"); return
    p = np.load(PREDS)
    ru, rb, lab, prob = p["review_user"], p["review_biz"], p["label"], p["prob_co"]

    # calibrated operating point: flag the top `true_rate` fraction by P(fake)
    true_rate = float((lab == 1).mean())
    k = int(round(true_rate * len(prob)))
    thresh = np.partition(prob, -k)[-k]
    flagged = prob >= thresh
    print(f"reviews {len(prob):,} | flagging top {true_rate:.1%} (P>={thresh:.3f}) -> {int(flagged.sum()):,} flagged")

    # extract rings from the GNN-flagged reviewers (NO labels used here)
    rings = rings_from(flagged, ru, rb)
    members = {u for r in rings for u in r}

    # validation: per-reviewer true-label tally over ALL labeled reviews
    user_lab = defaultdict(lambda: [0, 0])    # reviewer -> [#genuine, #fake]
    for i in range(len(ru)):
        if lab[i] in (0, 1):
            user_lab[int(ru[i])][int(lab[i])] += 1
    # businesses each flagged reviewer hit (for per-ring business count)
    biz_by_user = defaultdict(set)
    for u, b in zip(ru[flagged].tolist(), rb[flagged].tolist()):
        biz_by_user[u].add(b)

    tot_g = tot_f = revrs_fake = 0
    for u in members:
        g0, f1 = user_lab[u]; tot_g += g0; tot_f += f1
        if f1 > 0: revrs_fake += 1
    review_prec = tot_f / max(tot_g + tot_f, 1)
    revr_prec   = revrs_fake / max(len(members), 1)

    print("=" * 72)
    print("DEPLOYABLE COORDINATED RINGS  (reviewer-edge GNN, label-free extraction)")
    print("=" * 72)
    print(f"rings: {len(rings):,}   reviewers: {len(members):,}")
    print(f"VALIDATION vs full ground truth:")
    print(f"  members' reviews that are truly fake (review-precision): {review_prec:.1%}")
    print(f"  members with >=1 true fake review     (reviewer-precision): {revr_prec:.1%}")
    print(f"  enrichment over the {true_rate:.1%} base rate: {review_prec/true_rate:.1f}x")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ring_id", "n_reviewers", "n_businesses", "member_true_fake_rate", "reviewer_idxs"])
        for rid, m in enumerate(rings, 1):
            bizs = set().union(*(biz_by_user[u] for u in m)) if m else set()
            g0 = sum(user_lab[u][0] for u in m); f1 = sum(user_lab[u][1] for u in m)
            w.writerow([rid, len(m), len(bizs), round(f1 / max(g0 + f1, 1), 3),
                        "|".join(str(u) for u in sorted(m))])
    print(f"\nwrote {OUT} ({len(rings):,} rings)")

    print(f"\nTop 12 rings:")
    print(f"{'#':>2} {'revwrs':>6} {'bizs':>5} {'true-fake% of members':>22}")
    for i, m in enumerate(rings[:12], 1):
        bizs = set().union(*(biz_by_user[u] for u in m)) if m else set()
        g0 = sum(user_lab[u][0] for u in m); f1 = sum(user_lab[u][1] for u in m)
        print(f"{i:>2} {len(m):>6} {len(bizs):>5} {f1/max(g0+f1,1):>21.0%}")


if __name__ == "__main__":
    main()

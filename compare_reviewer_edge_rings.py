"""
compare_reviewer_edge_rings.py — does adding reviewer<->reviewer co-edges improve the
GNN's coordinated-ring detection?

Loads data/yelp_graph/yelp_gnn_reviewer_edges_preds.npz (prob_base = no co-edges,
prob_co = with co-edges; both trained identically, CE inverse-frequency) and, for EACH:

  * threshold to the true ~12.2% fake rate (matched calibration — we compare STRUCTURE,
    not the operating point),
  * extract rings (same union-find as extract_rings_gnn: >=2 shared businesses, cap 40,
    min 4 reviewers),
  * validate ring members against the FULL labels -> member-level precision.

So we can read off, side by side, whether the co-edges raised ring count / precision over
the ~56% no-edge baseline. Output: console table.

RUN (after uploading the notebook's npz back to data/yelp_graph/):
    python compare_reviewer_edge_rings.py
"""

import os
from collections import defaultdict

import numpy as np

from extract_rings_gnn import rings_from   # union-find ring extractor (MIN_SHARED=2, CAP=40, MIN_REVIEWERS=4)

GDIR = os.path.join(os.path.dirname(__file__), "data", "yelp_graph")
PREDS = os.path.join(GDIR, "yelp_gnn_reviewer_edges_preds.npz")


def threshold_to_rate(prob, rate):
    """Flag the top `rate` fraction by probability (matched-calibration operating point)."""
    k = int(round(rate * len(prob)))
    cut = np.partition(prob, -k)[-k]
    return prob >= cut


def evaluate(prob, ru, rb, lab, true_rate):
    is_fake = threshold_to_rate(prob, true_rate)
    rings = rings_from(is_fake, ru, rb)
    members = {u for r in rings for u in r}

    # per-reviewer true-label tally over ALL labeled reviews (full labels here)
    user_lab = defaultdict(lambda: [0, 0])     # reviewer -> [#genuine, #fake]
    for i in range(len(ru)):
        if lab[i] in (0, 1):
            user_lab[int(ru[i])][int(lab[i])] += 1

    tot_g = tot_f = 0
    revrs_truly_fake = 0
    for u in members:
        g0, f1 = user_lab[u]
        tot_g += g0; tot_f += f1
        if f1 > 0:
            revrs_truly_fake += 1
    review_prec = tot_f / max(tot_g + tot_f, 1)            # fraction of members' reviews that are truly fake
    revr_prec = revrs_truly_fake / max(len(members), 1)    # fraction of members with >=1 true fake review
    return len(rings), len(members), review_prec, revr_prec


def main():
    if not os.path.exists(PREDS):
        print(f"missing {PREDS}\nRun train_yelp_gnn_reviewer_edges.ipynb on Colab and put the npz here first.")
        return
    p = np.load(PREDS)
    ru, rb, lab = p["review_user"], p["review_biz"], p["label"]
    true_rate = float((lab == 1).mean())
    print(f"full labels: {len(lab):,} reviews | true fake rate {true_rate:.1%}")
    print(f"(both variants thresholded to that same rate -> only the graph STRUCTURE differs)\n")

    print(f"{'variant':26}{'rings':>7}{'reviewers':>11}{'review-prec':>13}{'reviewer-prec':>15}")
    for name, key in [("BASE (no co-edges)", "prob_base"), ("CO (reviewer<->reviewer)", "prob_co")]:
        if key not in p.files:
            print(f"{name:26}  (missing {key})"); continue
        nr, nm, rvp, rrp = evaluate(p[key], ru, rb, lab, true_rate)
        print(f"{name:26}{nr:>7}{nm:>11}{rvp:>12.0%}{rrp:>14.0%}")
    print(f"\n(baseline reference from the focal sweep: ~24-27 rings, ~55-56% precision)")


if __name__ == "__main__":
    main()

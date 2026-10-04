# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
extract_rings_gnn.py — coordinated-ring extraction driven by the GNN's PREDICTIONS
(no ground-truth labels), then validated against the labels.

This is the *deployable* pipeline and the actual point of the GNN: in production there
are no labels, so the GNN predicts which reviews are fake, and rings are extracted from
the GNN-FLAGGED reviewers. We then check the ground truth to see whether those rings are
real (i.e. whether the flagged reviewers are actually fake).

Uses data/yelp_graph/graph.npz (co-review structure) + data/yelp_graph/yelp_gnn_preds.npz
(per-review GNN prediction + the true label), aligned by review-node index.

RUN:  python extract_rings_gnn.py
"""

import os
from collections import defaultdict
from itertools import combinations

import numpy as np

GDIR = os.path.join(os.path.dirname(__file__), "data", "yelp_graph")
# operating point (chosen from the recall/precision sweep): 37 rings / 304 reviewers @ 88% precision
THRESH = 0.70           # flag a review fake if GNN P(fake) >= THRESH (calibrates the 50/50-trained over-prediction)
MIN_SHARED, CAP, MIN_REVIEWERS = 2, 40, 4
OUT_CSV = os.path.join(GDIR, "yelp_rings_gnn.csv")


def rings_from(is_fake, review_user, review_biz):
    """Link reviewers sharing >= MIN_SHARED businesses both flagged fake; return components."""
    biz_revrs = defaultdict(set)
    for i in range(len(review_user)):
        if is_fake[i]:
            biz_revrs[int(review_biz[i])].add(int(review_user[i]))
    pair = defaultdict(int)
    for rs in biz_revrs.values():
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
        if cnt >= MIN_SHARED:
            ra, rb = find(a), find(c)
            if ra != rb: parent[ra] = rb
    comp = defaultdict(list)
    for u in list(parent):
        comp[find(u)].append(u)
    return sorted((m for m in comp.values() if len(m) >= MIN_REVIEWERS), key=len, reverse=True)


def main():
    g = np.load(os.path.join(GDIR, "graph.npz"))
    p = np.load(os.path.join(GDIR, "yelp_gnn_preds.npz"))
    ru, rb = g["review_user"], g["review_biz"]
    pred, lab = p["pred"], p["label"]   # pred: GNN fake/genuine (all nodes); lab: -1 unlabeled / 0 / 1

    # per-reviewer true-label tally (for validation, on labeled reviews only)
    user_lab = defaultdict(lambda: [0, 0])   # reviewer -> [#genuine, #fake] among labeled
    for i in range(len(ru)):
        if lab[i] in (0, 1):
            user_lab[int(ru[i])][int(lab[i])] += 1

    # 1) GNN-driven rings (NO labels used) — confidence-thresholded to calibrate over-prediction
    gnn_rings = rings_from(p["prob_fake"] >= THRESH, ru, rb)
    gnn_members = {u for r in gnn_rings for u in r}

    # save the GNN-found rings (with per-ring business count + true-fake validation)
    flagged = p["prob_fake"] >= THRESH          # vectorized; do NOT index the npz array in a loop
    biz_by_user = defaultdict(set)
    for u, b in zip(ru[flagged].tolist(), rb[flagged].tolist()):
        biz_by_user[u].add(b)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["ring_id", "n_reviewers", "n_businesses", "member_true_fake_rate", "reviewer_idxs"])
        for rid, members in enumerate(gnn_rings, 1):
            bizs = set().union(*(biz_by_user[u] for u in members)) if members else set()
            g0 = sum(user_lab[u][0] for u in members); f1 = sum(user_lab[u][1] for u in members)
            w.writerow([rid, len(members), len(bizs), round(f1 / max(g0 + f1, 1), 3),
                        "|".join(str(u) for u in sorted(members))])
    print(f"wrote {OUT_CSV} ({len(gnn_rings)} GNN-found rings)")

    # 2) ground-truth rings (labels used) — reference
    gt_rings = rings_from(lab == 1, ru, rb)
    gt_members = {u for r in gt_rings for u in r}

    # validation: of GNN-ring members' LABELED reviews, what fraction are truly fake?
    tot_g = tot_f = 0
    for u in gnn_members:
        g0, f1 = user_lab[u]; tot_g += g0; tot_f += f1
    member_fake_rate = tot_f / max(tot_g + tot_f, 1)
    # how many GNN-ring reviewers are also flagged by ground truth as fake (>=1 fake review)?
    gnn_members_truly_fake = sum(1 for u in gnn_members if user_lab[u][1] > 0)
    overlap = len(gnn_members & gt_members)

    print("="*72)
    print("GNN-DRIVEN COORDINATED RINGS  (extracted from GNN predictions, no labels)")
    print("="*72)
    print(f"GNN-found rings: {len(gnn_rings):,}   reviewers in them: {len(gnn_members):,}")
    print(f"\nVALIDATION against ground truth (the GNN never saw these at ring-time):")
    print(f"  GNN-ring members' labeled reviews that are TRULY fake: {member_fake_rate:.1%}")
    print(f"  GNN-ring reviewers with >=1 true fake review:          "
          f"{gnn_members_truly_fake}/{len(gnn_members)} ({gnn_members_truly_fake/max(len(gnn_members),1):.1%})")
    print(f"  reviewers in BOTH GNN-rings and ground-truth-rings:    {overlap}")
    print(f"\nReference: ground-truth-label rings = {len(gt_rings):,} rings / {len(gt_members):,} reviewers")

    print(f"\nTop GNN-found rings:")
    print(f"{'#':>2} {'revwrs':>6} {'true-fake% of members':>22}")
    for i, members in enumerate(gnn_rings[:10], 1):
        g0 = sum(user_lab[u][0] for u in members); f1 = sum(user_lab[u][1] for u in members)
        fr = f1 / max(g0 + f1, 1)
        print(f"{i:>2} {len(members):>6} {fr:>21.0%}")


if __name__ == "__main__":
    main()

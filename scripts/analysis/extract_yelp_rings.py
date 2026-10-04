# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
extract_yelp_rings.py — surface COORDINATED FAKE-REVIEW RINGS (advisor's GNN goal).

The GNN classifies each review fake/genuine, but the advisor asked specifically for
*rings* — groups of reviewers acting together. This script extracts them structurally
from the full raw Yelp data (YelpZip + YelpNYC + YelpChi, ~1.03M reviews):

  * Read metadata only (user, business, date, rating, fake-label) — no text, so it's fast.
  * A "ring" = a connected cluster of FAKE reviewers linked by **co-attacking the same
    business within a short window** (genuine reviewers don't cluster in time; coordinated
    crews hit a target in bursts). Union-Find over reviewers; components of size >= MIN.
  * Report the largest rings + how concentrated fake activity is in rings.

This is the relational signal the per-review text/behavioral models (Model A, Model B)
structurally cannot produce. Output: console summary + data/yelp_rings.csv (top rings).

RUN:  python extract_yelp_rings.py
"""

import os
import csv
from collections import defaultdict

import build_yelp_multimodal_features as M   # raw-file paths + _day_ordinal (has __main__ guard)

MIN_SHARED    = 3    # link two fake reviewers if they jointly fake-reviewed >= this many businesses
CAP           = 30   # ignore mega-target businesses (> CAP fake reviewers) — co-review there is noise
MIN_REVIEWERS = 4    # minimum reviewers to call a cluster a "ring"
OUT_CSV       = os.path.join(M.DATA_DIR, "yelp_rings.csv")


def _read(records):
    """records: list of (user, biz, day, rating, fake)."""
    def zip_style(path, prefix):
        n = 0
        for line in open(path, encoding="utf-8", errors="ignore"):
            m = line.rstrip("\n").split("\t")
            if len(m) < 5:
                continue
            try: rating = float(m[2])
            except ValueError: continue
            records.append((f"{prefix}_{m[0].strip()}", f"{prefix}_{m[1].strip()}",
                            M._day_ordinal(m[4].strip(), "%Y-%m-%d"), rating, m[3].strip() == "-1"))  # -1 = fake
            n += 1
        print(f"  {prefix}: {n:,}")

    def chi_style(path, prefix):
        n = 0
        for line in open(path, encoding="utf-8", errors="ignore"):
            p = line.split()
            if len(p) < 9:
                continue
            try: rating = float(p[8])
            except ValueError: continue
            # date, review_id, reviewer_id, product_id, label(Y/N), ..., rating
            records.append((f"{prefix}_{p[2]}", f"{prefix}_{p[3]}",
                            M._day_ordinal(p[0], "%m/%d/%Y"), rating, p[4].strip().upper() == "Y"))
            n += 1
        print(f"  {prefix}: {n:,}")

    zip_style(M.YELPZIP_META, "zip")
    zip_style(M.YELPNYC_META, "nyc")
    chi_style(M.CHI_RES_META, "chi_res")
    chi_style(M.CHI_HOT_META, "chi_hot")


def main():
    print("reading raw Yelp metadata (full data) ...")
    recs = []
    _read(recs)
    print(f"total reviews: {len(recs):,}")

    # per-user review history (for ring stats)
    user_recs = defaultdict(list)
    for r in recs:
        user_recs[r[0]].append(r)
    n_fake_reviewers = sum(1 for u, rs in user_recs.items() if any(x[4] for x in rs))

    # Union-Find over reviewers, linking those who co-attacked a business within W_DAYS
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # link two fake reviewers iff they jointly fake-reviewed >= MIN_SHARED of the SAME
    # businesses (genuine pairs rarely share several specific fake targets — this is the
    # coordination signal). Mega-target businesses (> CAP fake reviewers) are skipped:
    # co-review there is noise, not a crew.
    from itertools import combinations
    biz_revrs = defaultdict(set)        # biz -> {fake reviewers}
    for u, b, day, rating, fake in recs:
        if fake:
            biz_revrs[b].add(u)
    pair = defaultdict(int)             # (reviewer_a, reviewer_b) -> # shared fake-target businesses
    for b, rs in biz_revrs.items():
        if 2 <= len(rs) <= CAP:
            for a, c in combinations(sorted(rs), 2):
                pair[(a, c)] += 1
    for (a, c), cnt in pair.items():
        if cnt >= MIN_SHARED:
            union(a, c)

    comp = defaultdict(list)
    for u in list(parent):
        comp[find(u)].append(u)
    rings = sorted((m for m in comp.values() if len(m) >= MIN_REVIEWERS), key=len, reverse=True)

    reviewers_in_rings = sum(len(r) for r in rings)
    print(f"\n{'='*70}")
    print(f"COORDINATED RINGS  (link: >= {MIN_SHARED} shared fake-target businesses; min size {MIN_REVIEWERS})")
    print(f"{'='*70}")
    print(f"rings found: {len(rings):,}   |   reviewers in rings: {reviewers_in_rings:,}   "
          f"({reviewers_in_rings/max(n_fake_reviewers,1):.1%} of all fake reviewers)")

    def ring_stats(members):
        ms = set(members)
        allr = [x for u in members for x in user_recs[u]]
        fakes = [x for x in allr if x[4]]
        bizs = {x[1] for x in fakes}
        days = [x[2] for x in fakes if x[2] is not None]
        ratings = [x[3] for x in fakes]
        span = (max(days) - min(days)) if days else 0
        return {
            "reviewers": len(ms), "businesses": len(bizs), "fake_reviews": len(fakes),
            "total_reviews": len(allr),
            "fake_rate": len(fakes) / max(len(allr), 1),
            "span_days": span,
            "avg_rating": (sum(ratings) / len(ratings)) if ratings else 0,
            "pct_extreme": sum(1 for r in ratings if r in (1.0, 5.0)) / max(len(ratings), 1),
        }

    print(f"\nTop 12 rings:")
    print(f"{'#':>2} {'revwrs':>6} {'bizs':>5} {'fakeRv':>6} {'fake%':>6} {'spanD':>6} {'avgRt':>5} {'extreme%':>8}")
    rows = []
    for i, members in enumerate(rings[:12], 1):
        s = ring_stats(members)
        print(f"{i:>2} {s['reviewers']:>6} {s['businesses']:>5} {s['fake_reviews']:>6} "
              f"{s['fake_rate']:>5.0%} {s['span_days']:>6} {s['avg_rating']:>5.1f} {s['pct_extreme']:>7.0%}")
    # save all rings
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ring_id", "n_reviewers", "n_businesses", "fake_reviews", "total_reviews",
                    "fake_rate", "span_days", "avg_rating", "pct_extreme", "reviewer_ids"])
        for i, members in enumerate(rings, 1):
            s = ring_stats(members)
            w.writerow([i, s["reviewers"], s["businesses"], s["fake_reviews"], s["total_reviews"],
                        round(s["fake_rate"], 3), s["span_days"], round(s["avg_rating"], 2),
                        round(s["pct_extreme"], 3), "|".join(sorted(members)[:200])])
    print(f"\nwrote {OUT_CSV} ({len(rings):,} rings)")


if __name__ == "__main__":
    main()

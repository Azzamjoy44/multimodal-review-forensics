"""
csv_behavior.py — Tier 3-lite: within-CSV reviewer-behaviour heuristics for the
Dashboard upload box.

Unlike the multi-modal Model B (which needs platform-scale reviewer history that an
upload can't provide), these signals are computed WITHIN the uploaded batch and are
reliable — they flag classic coordinated/astroturf patterns that ARE visible in a
single upload:
  * a reviewer posting many reviews in one upload (prolific),
  * a reviewer flooding identical or all-extreme (1/5) ratings,
  * a reviewer bursting many reviews on the same day.

Activates only when the CSV has a user column (rating/date optional, enable more checks).
Each review by a flagged reviewer is tagged so the frontend can surface it.
"""

from datetime import datetime
from collections import defaultdict

PROLIFIC = 5    # >= this many reviews by one user in the upload
MONOTONE = 4    # >= this many reviews to judge rating monotony / all-extreme
BURST    = 3    # >= this many reviews by one user on the same day

_ALIASES = {
    "user":   ["user_id", "userid", "user", "reviewer_id", "reviewer", "author", "author_id"],
    "rating": ["rating", "stars", "star", "score", "rating_value"],
    "date":   ["date", "time", "timestamp", "review_date", "created_at", "datetime"],
}


def detect_columns(header):
    """header = list of column names. Returns {role: index} (user required) or None."""
    h = [str(c).strip().lower() for c in header]
    found = {}
    for role, aliases in _ALIASES.items():
        for a in aliases:
            if a in h:
                found[role] = h.index(a); break
    return found if "user" in found else None


def to_day(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S", "%m-%d-%Y"):
        try:
            return datetime.strptime(s[:len(fmt) + 2], fmt).date().toordinal()
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromisoformat(s).date().toordinal()
    except Exception:
        return None


def analyze_reviewers(users, ratings=None, days=None):
    """Per-review behaviour tag based on within-batch reviewer patterns.
    Returns list of {suspicious: bool, flags: [str], detail: str}, one per review."""
    n = len(users)
    by_user = defaultdict(list)
    for i, u in enumerate(users):
        u = str(u).strip()
        if u:
            by_user[u].append(i)

    flags_per = [[] for _ in range(n)]
    for u, idxs in by_user.items():
        c = len(idxs)
        ufl = []
        if c >= PROLIFIC:
            ufl.append(f"posted {c} reviews in this upload")
        if ratings and c >= MONOTONE:
            rs = [ratings[i] for i in idxs if ratings[i] is not None]
            if len(rs) >= MONOTONE:
                if len(set(rs)) == 1:
                    ufl.append(f"all {len(rs)} reviews rated {rs[0]:g}★")
                elif all(r in (1.0, 5.0) for r in rs):
                    ufl.append(f"all {len(rs)} reviews extreme (1/5★)")
        if days and c >= BURST:
            dc = defaultdict(int)
            for i in idxs:
                if days[i] is not None:
                    dc[days[i]] += 1
            if dc and max(dc.values()) >= BURST:
                ufl.append(f"{max(dc.values())} reviews on the same day")
        if ufl:
            for i in idxs:
                flags_per[i] = ufl

    return [{"suspicious": bool(f), "flags": f, "detail": "; ".join(f)} for f in flags_per]


def from_rows(data_rows, colmap):
    """Convenience: extract user/rating/day columns from raw CSV rows and analyze."""
    def cell(r, idx):
        return r[idx] if (idx is not None and idx < len(r)) else ""
    users = [cell(r, colmap.get("user")) for r in data_rows]
    ratings = None
    if "rating" in colmap:
        ratings = []
        for r in data_rows:
            try:
                ratings.append(float(str(cell(r, colmap["rating"])).strip()))
            except (ValueError, TypeError):
                ratings.append(None)
    days = None
    if "date" in colmap:
        days = [to_day(cell(r, colmap["date"])) for r in data_rows]
    return analyze_reviewers(users, ratings, days)

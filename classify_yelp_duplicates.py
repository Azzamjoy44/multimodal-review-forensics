"""
classify_yelp_duplicates.py — what does the text-only dedup actually REMOVE?
Splits every removed-by-text-dedup row into three buckets, comparing it to the kept
(first) occurrence of the same normalized text:
  * cross_dataset  — different source dataset (NYC<->Zip overlap): same real review, other ID scheme
  * exact_dup      — SAME source & SAME (reviewer,business): one real action counted twice (safe to drop)
  * intra_template — SAME source, DIFFERENT (reviewer/business): a distinct review = coordinated near-dupe (SIGNAL, keep)
RUN: python classify_yelp_duplicates.py
"""
from collections import defaultdict
import build_yelp_multimodal_features as M
import build_yelp_multimodal_features_full as MF

recs = []
MF.read_zip(M.YELPZIP_META, M.YELPZIP_REV, "zip", recs)
MF.read_zip(M.YELPNYC_META, M.YELPNYC_REV, "nyc", recs)
MF.read_chi(M.CHI_RES_META, M.CHI_RES_REV, "chi_res", recs)
MF.read_chi(M.CHI_HOT_META, M.CHI_HOT_REV, "chi_hot", recs)
print(f"raw recs: {len(recs):,}")

first = {}                       # norm -> first (kept) record
buckets = defaultdict(int)
for r in recs:
    n = r["norm"]
    if n not in first:
        first[n] = r            # kept
        continue
    k = first[n]                 # this row is REMOVED; compare to the kept first
    if r["source"] != k["source"]:
        buckets["cross_dataset"] += 1
    elif r["user"] == k["user"] and r["biz"] == k["biz"]:
        buckets["exact_dup"] += 1
    else:
        buckets["intra_template"] += 1

removed = sum(buckets.values())
print(f"\nremoved by text-dedup: {removed:,}")
for b in ("cross_dataset", "exact_dup", "intra_template"):
    print(f"  {b:<16} {buckets[b]:>9,}  ({buckets[b]/removed:.1%})")
print("\nintra_template = coordinated near-dupes across DIFFERENT businesses = fraud SIGNAL (should keep)")
print("exact_dup      = same reviewer+business+text twice = double-count (safe to drop)")
print("cross_dataset  = NYC<->Zip same review under other IDs (doesn't touch the kept entity's features)")

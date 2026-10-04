"""
build_yelp_graph.py
-------------------
Builds the heterogeneous graph for the **coordinated-fake-review-ring GNN** (advisor's
suggestion). Reuses the raw-file readers + aggregate logic from
`build_yelp_multimodal_features.py` (imported, not duplicated).

Graph design (see TODO.md → GNN thread):
  * Tripartite, heterogeneous: nodes = {reviewer, review, business}. A review connects
    to its reviewer and its business; rings emerge as reviewers sharing businesses.
  * FULL graph, PARTIAL labels: every raw review (~1.03M across YelpZip/NYC/Chi) is a
    review node, so coordination structure stays intact. Labels + train/val/test masks
    are set only on the ~178k reviews present in `yelp_split.csv` (the SAME reviews +
    split as Model B / the text-only baseline → clean head-to-head). Semi-supervised:
    the GNN message-passes over the full graph, loss is computed only on labeled nodes.
  * The three regions have separate user/business ID spaces (IDs are region-prefixed),
    so this is three disjoint sub-graphs in one file — fine for a GNN.
  * Node features (behavioral + structure; NO text embedding in this first version):
      review   (6): rating, is_extreme, word_len, char_len, rating_dev_from_biz, abs_dev
      reviewer (8): review_count, avg_rating, rating_std, frac_positive, frac_extreme,
                    is_singleton, max_reviews_per_day, reviews_per_day
      business (3): review_count, avg_rating, rating_std
    Aggregates are computed over the FULL raw data (real history), as in the features pipeline.

Output: data/yelp_graph/graph.npz (+ meta.json). Library-agnostic (numpy only) so the
Colab training notebook loads it into PyTorch Geometric / DGL. Edges are reconstructed
there from review_user / review_biz (each review -> its reviewer + business).

RUN:  python build_yelp_graph.py        # ~1-2 GB RAM transiently; don't run alongside other big jobs
"""

import os
import csv
import json
from collections import defaultdict

import numpy as np

import build_yelp_multimodal_features as M   # raw readers, aggregate helpers, paths (has __main__ guard)

OUT_DIR = os.path.join(M.DATA_DIR, "yelp_graph")
SPLIT_CODE = {"train": 1, "val": 2, "test": 3}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) split lookup: normalized text -> (label, split)
    split_lookup = {}
    with open(M.SPLIT_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            split_lookup[M.normalize(row["text"])] = (int(row["label"]), row["split"])
    M.log(f"yelp_split.csv: {len(split_lookup):,} unique normalized texts")

    # 2) read all raw reviews (full data)
    M.log("reading raw Yelp files (full data) ...")
    records = []
    M.read_zip_style(M.YELPZIP_META, M.YELPZIP_REV, "zip",     records)
    M.read_zip_style(M.YELPNYC_META, M.YELPNYC_REV, "nyc",     records)
    M.read_chi_style(M.CHI_RES_META, M.CHI_RES_REV, "chi_res", records)
    M.read_chi_style(M.CHI_HOT_META, M.CHI_HOT_REV, "chi_hot", records)
    M.log(f"total raw reviews (review nodes): {len(records):,}")

    # 3) aggregates over full data (same definitions as the features pipeline)
    M.log("building reviewer / business aggregates ...")
    user_ratings, user_days, biz_ratings = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        user_ratings[r["user"]].append(r["rating"])
        if r["day"] is not None:
            user_days[r["user"]].append(r["day"])
        biz_ratings[r["biz"]].append(r["rating"])

    def max_per_day(days):
        if not days:
            return 0
        c = defaultdict(int)
        for d in days:
            c[d] += 1
        return max(c.values())

    user_count = {u: len(v) for u, v in user_ratings.items()}
    user_avg   = {u: sum(v) / len(v) for u, v in user_ratings.items()}
    user_std   = {u: M._std(v) for u, v in user_ratings.items()}
    user_fpos  = {u: sum(1 for x in v if x >= 4) / len(v) for u, v in user_ratings.items()}
    user_fext  = {u: sum(1 for x in v if x in (1.0, 5.0)) / len(v) for u, v in user_ratings.items()}
    user_burst = {u: max_per_day(d) for u, d in user_days.items()}
    user_span  = {u: (max(d) - min(d) + 1) if d else 1 for u, d in user_days.items()}
    biz_count  = {b: len(v) for b, v in biz_ratings.items()}
    biz_avg    = {b: sum(v) / len(v) for b, v in biz_ratings.items()}
    biz_std    = {b: M._std(v) for b, v in biz_ratings.items()}

    # 4) node id maps (region-prefixed IDs are already disjoint across datasets)
    users, bizs = {}, {}
    for r in records:
        users.setdefault(r["user"], len(users))
        bizs.setdefault(r["biz"], len(bizs))
    Nr, Nu, Nb = len(records), len(users), len(bizs)

    # 5) review nodes: features + edges-to-user/biz + labels/splits (deduped by norm like the features pipeline)
    rev_feat  = np.zeros((Nr, 6), dtype=np.float32)
    rev_user  = np.zeros(Nr, dtype=np.int64)
    rev_biz   = np.zeros(Nr, dtype=np.int64)
    rev_label = np.full(Nr, -1, dtype=np.int64)     # -1 = unlabeled (structural-only node)
    rev_split = np.zeros(Nr, dtype=np.int64)        # 0 none / 1 train / 2 val / 3 test
    seen = set()
    for i, r in enumerate(records):
        b = r["biz"]
        rev_feat[i] = (
            r["rating"],
            1.0 if r["rating"] in (1.0, 5.0) else 0.0,
            len(r["text"].split()),
            len(r["text"]),
            round(r["rating"] - biz_avg[b], 4),
            round(abs(r["rating"] - biz_avg[b]), 4),
        )
        rev_user[i] = users[r["user"]]
        rev_biz[i]  = bizs[b]
        hit = split_lookup.get(r["norm"])
        if hit is not None and r["norm"] not in seen:   # first raw occurrence becomes the labeled node
            seen.add(r["norm"])
            rev_label[i] = hit[0]
            rev_split[i] = SPLIT_CODE[hit[1]]

    # 6) reviewer + business node feature matrices
    usr_feat = np.zeros((Nu, 8), dtype=np.float32)
    for u, idx in users.items():
        span = user_span.get(u, 1)
        usr_feat[idx] = (
            user_count[u], round(user_avg[u], 4), round(user_std[u], 4),
            round(user_fpos[u], 4), round(user_fext[u], 4),
            1.0 if user_count[u] == 1 else 0.0,
            user_burst.get(u, 0), round(user_count[u] / max(span, 1), 5),
        )
    biz_feat = np.zeros((Nb, 3), dtype=np.float32)
    for b, idx in bizs.items():
        biz_feat[idx] = (biz_count[b], round(biz_avg[b], 4), round(biz_std[b], 4))

    # 7) save (numpy-only; Colab builds the PyG/DGL graph from these)
    np.savez_compressed(
        os.path.join(OUT_DIR, "graph.npz"),
        review_feat=rev_feat, review_user=rev_user, review_biz=rev_biz,
        review_label=rev_label, review_split=rev_split,
        reviewer_feat=usr_feat, business_feat=biz_feat,
    )
    lab = rev_label[rev_label >= 0]
    meta = {
        "n_review_nodes": int(Nr), "n_reviewer_nodes": int(Nu), "n_business_nodes": int(Nb),
        "n_edges_review_reviewer": int(Nr), "n_edges_review_business": int(Nr),
        "labeled_reviews": int((rev_label >= 0).sum()),
        "label_balance": {"genuine_0": int((lab == 0).sum()), "fake_1": int((lab == 1).sum())},
        "split_counts": {"train": int((rev_split == 1).sum()),
                         "val":   int((rev_split == 2).sum()),
                         "test":  int((rev_split == 3).sum())},
        "review_features": ["rating", "is_extreme", "word_len", "char_len",
                            "rating_dev_from_biz", "abs_rating_dev_from_biz"],
        "reviewer_features": ["review_count", "avg_rating", "rating_std", "frac_positive",
                              "frac_extreme", "is_singleton", "max_reviews_per_day", "reviews_per_day"],
        "business_features": ["review_count", "avg_rating", "rating_std"],
        "note": "tripartite reviewer<->review<->business; full graph, partial labels; "
                "labeled/test set == yelp_split (same as Model B). No text features (v1).",
    }
    json.dump(meta, open(os.path.join(OUT_DIR, "meta.json"), "w"), indent=2)

    M.log(f"\nwrote {os.path.join(OUT_DIR, 'graph.npz')}")
    M.log(f"  review nodes:   {Nr:,}  (labeled {meta['labeled_reviews']:,})")
    M.log(f"  reviewer nodes: {Nu:,}")
    M.log(f"  business nodes: {Nb:,}")
    M.log(f"  label balance:  {meta['label_balance']}")
    M.log(f"  split counts:   {meta['split_counts']}")


if __name__ == "__main__":
    main()

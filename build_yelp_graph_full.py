"""
build_yelp_graph_full.py — full-label graph for the focal-loss GNN experiment.

Unlike build_yelp_graph.py (which labels only the 178k balanced `yelp_split` subset and
leaves ~857k reviews UNLABELED), this labels **every one of the ~1.03M reviews** from
Yelp's raw spam flags — the true ~13%-fake distribution. The goal: train the GNN on the
real distribution (with focal loss to handle the imbalance) so its predictions are
**calibrated** (no more 38% over-prediction) and *every* reviewer is a training signal —
which should help coordinated-ring recall + give fully-verifiable rings.

Split: the existing `yelp_split` test/val reviews stay test/val (so classification is
comparable to the other models); everything else is train. Labels come from the raw
metadata (zip/nyc col 3: '1'=fake / '-1'=genuine; chi col 4: 'Y'=fake / 'N'=genuine).

Output: data/yelp_graph/graph_full.npz (+ graph_full_meta.json). Same tripartite layout
as graph.npz (reviewer<->review<->business); load it the same way in the notebook.

RUN:  python build_yelp_graph_full.py
"""

import os
import csv
import json
from collections import defaultdict

import numpy as np

import build_yelp_multimodal_features as M   # paths, normalize, _day_ordinal, _std

OUT_DIR = os.path.join(M.DATA_DIR, "yelp_graph")
SPLIT_CODE = {"train": 1, "val": 2, "test": 3}


def read_zip(meta, rev, prefix, recs):
    if not os.path.exists(meta):
        print(f"  {prefix}: missing"); return
    n = 0
    with open(meta, encoding="utf-8", errors="ignore") as mf, \
         open(rev, encoding="utf-8", errors="ignore") as rf:
        for ml, rl in zip(mf, rf):
            m = ml.strip().split("\t"); r = rl.strip().split("\t")
            if len(m) < 5 or len(r) < 4:
                continue
            text = "\t".join(r[3:]).strip()
            if not text:
                continue
            try: rating = float(m[2])
            except ValueError: continue
            recs.append({"user": f"{prefix}_{m[0].strip()}", "biz": f"{prefix}_{m[1].strip()}",
                         "rating": rating, "day": M._day_ordinal(m[4].strip(), "%Y-%m-%d"),
                         "text": text, "label": 1 if m[3].strip() == "-1" else 0,  # -1 = fake (filtered)
                         "norm": M.normalize(text)})
            n += 1
    print(f"  {prefix}: {n:,}")


def read_chi(meta, rev, prefix, recs):
    if not os.path.exists(meta):
        print(f"  {prefix}: missing"); return
    n = 0
    with open(meta, encoding="utf-8", errors="ignore") as mf, \
         open(rev, encoding="utf-8", errors="ignore") as rf:
        for ml, rl in zip(mf, rf):
            p = ml.strip().split(); text = rl.strip()
            if len(p) < 9 or not text:
                continue
            try: rating = float(p[8])
            except ValueError: continue
            # cols: date[0] review_id[1] reviewer_id[2] product_id[3] label[4] ... rating[8]
            recs.append({"user": f"{prefix}_{p[2]}", "biz": f"{prefix}_{p[3]}",
                         "rating": rating, "day": M._day_ordinal(p[0], "%m/%d/%Y"),
                         "text": text, "label": 1 if p[4].strip().upper() == "Y" else 0,
                         "norm": M.normalize(text)})
            n += 1
    print(f"  {prefix}: {n:,}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # yelp_split split assignment (norm -> 'train'/'val'/'test'); keeps test/val comparable
    split_of = {}
    with open(M.SPLIT_CSV, encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            split_of[M.normalize(row["text"])] = row["split"]
    print(f"yelp_split: {len(split_of):,} norms")

    print("reading raw Yelp (meta+rev, full data) ...")
    recs = []
    read_zip(M.YELPZIP_META, M.YELPZIP_REV, "zip",     recs)
    read_zip(M.YELPNYC_META, M.YELPNYC_REV, "nyc",     recs)
    read_chi(M.CHI_RES_META, M.CHI_RES_REV, "chi_res", recs)
    read_chi(M.CHI_HOT_META, M.CHI_HOT_REV, "chi_hot", recs)
    print(f"total reviews: {len(recs):,}")

    # CROSS-DATASET text-dedup: YelpNYC is ~99% a subset of YelpZip (~353k shared review
    # texts under separate id schemes), so the raw union double-counts. Keep-first (zip is
    # read first) -> ALL YelpZip + ALL YelpChi + only the NYC-UNIQUE reviews. Done BEFORE the
    # aggregates so reviewer/business counts aren't inflated by the duplicates.
    seen, deduped = set(), []
    for r in recs:
        if r["norm"] in seen:
            continue
        seen.add(r["norm"]); deduped.append(r)
    print(f"after cross-dataset text-dedup: {len(recs):,} -> {len(deduped):,} unique")
    recs = deduped

    # aggregates over full data (same definitions as build_yelp_graph.py)
    print("building reviewer / business aggregates ...")
    ur, ud, br = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in recs:
        ur[r["user"]].append(r["rating"])
        if r["day"] is not None: ud[r["user"]].append(r["day"])
        br[r["biz"]].append(r["rating"])
    def mpd(days):
        if not days: return 0
        c = defaultdict(int)
        for d in days: c[d] += 1
        return max(c.values())
    ucnt = {u: len(v) for u, v in ur.items()}
    uavg = {u: sum(v)/len(v) for u, v in ur.items()}
    ustd = {u: M._std(v) for u, v in ur.items()}
    ufpos = {u: sum(1 for x in v if x >= 4)/len(v) for u, v in ur.items()}
    ufext = {u: sum(1 for x in v if x in (1.0, 5.0))/len(v) for u, v in ur.items()}
    uburst = {u: mpd(d) for u, d in ud.items()}
    uspan = {u: (max(d)-min(d)+1) if d else 1 for u, d in ud.items()}
    bcnt = {b: len(v) for b, v in br.items()}
    bavg = {b: sum(v)/len(v) for b, v in br.items()}
    bstd = {b: M._std(v) for b, v in br.items()}

    users, bizs = {}, {}
    for r in recs:
        users.setdefault(r["user"], len(users))
        bizs.setdefault(r["biz"], len(bizs))
    Nr, Nu, Nb = len(recs), len(users), len(bizs)

    rev_feat = np.zeros((Nr, 6), dtype=np.float32)
    rev_user = np.zeros(Nr, dtype=np.int64)
    rev_biz  = np.zeros(Nr, dtype=np.int64)
    rev_label = np.zeros(Nr, dtype=np.int64)     # EVERY review labeled (0/1) from raw
    rev_split = np.ones(Nr, dtype=np.int64)       # default train(1); test/val from yelp_split
    rev_day   = np.full(Nr, -1, dtype=np.int64)   # day ordinal per review (-1 if unparseable); for temporal co-edges
    for i, r in enumerate(recs):
        b = r["biz"]
        rev_feat[i] = (r["rating"], 1.0 if r["rating"] in (1.0, 5.0) else 0.0,
                       len(r["text"].split()), len(r["text"]),
                       round(r["rating"]-bavg[b], 4), round(abs(r["rating"]-bavg[b]), 4))
        rev_user[i] = users[r["user"]]; rev_biz[i] = bizs[b]
        rev_label[i] = r["label"]
        rev_split[i] = SPLIT_CODE.get(split_of.get(r["norm"], "train"), 1)
        if r["day"] is not None: rev_day[i] = r["day"]

    usr_feat = np.zeros((Nu, 8), dtype=np.float32)
    for u, idx in users.items():
        span = uspan.get(u, 1)
        usr_feat[idx] = (ucnt[u], round(uavg[u], 4), round(ustd[u], 4), round(ufpos[u], 4),
                         round(ufext[u], 4), 1.0 if ucnt[u] == 1 else 0.0,
                         uburst.get(u, 0), round(ucnt[u]/max(span, 1), 5))
    biz_feat = np.zeros((Nb, 3), dtype=np.float32)
    for b, idx in bizs.items():
        biz_feat[idx] = (bcnt[b], round(bavg[b], 4), round(bstd[b], 4))

    np.savez_compressed(os.path.join(OUT_DIR, "graph_full.npz"),
        review_feat=rev_feat, review_user=rev_user, review_biz=rev_biz,
        review_label=rev_label, review_split=rev_split, review_day=rev_day,
        reviewer_feat=usr_feat, business_feat=biz_feat)
    meta = {
        "n_review_nodes": int(Nr), "n_reviewer_nodes": int(Nu), "n_business_nodes": int(Nb),
        "label_balance": {"genuine_0": int((rev_label == 0).sum()), "fake_1": int((rev_label == 1).sum())},
        "fake_rate": round(float(rev_label.mean()), 4),
        "split_counts": {"train": int((rev_split == 1).sum()),
                         "val": int((rev_split == 2).sum()),
                         "test": int((rev_split == 3).sum())},
        "note": "ALL reviews labeled from raw Yelp flags (true distribution); test/val = yelp_split, "
                "rest = train. For focal-loss GNN. Same tripartite layout as graph.npz.",
    }
    json.dump(meta, open(os.path.join(OUT_DIR, "graph_full_meta.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'graph_full.npz')}")
    print(f"  reviews {Nr:,} | reviewers {Nu:,} | businesses {Nb:,}")
    print(f"  fake rate (full, true distribution): {meta['fake_rate']:.1%}  -> {meta['label_balance']}")
    print(f"  split: {meta['split_counts']}")


if __name__ == "__main__":
    main()

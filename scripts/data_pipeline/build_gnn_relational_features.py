# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
build_gnn_relational_features.py — per-reviewer RELATIONAL features from the reviewer co-review
graph, to fuse into Model B (chase the GNN-vs-ModelB oracle). Output aligns row-for-row with
yelp_multimodal_features_full.csv / the GNN temporal preds (graph_full review i == features row i).

Features (per reviewer, mapped to each of their reviews):
  STRUCTURAL (label-free -> safe for all splits): degree, component_size (structural ring size),
    core_number (k-core depth), clustering, temporal_degree (temporal-burst co-edge degree)
  LABEL-AWARE (TRAIN labels only -> no test leak): reviewer_train_fraud (fraction of this reviewer's
    TRAIN reviews flagged fake), neighbor_fraud_rate (mean reviewer_train_fraud over co-edge neighbors)

Output: data/yelp_graph/relational_features.npz  (rel [Nr,7] float32 + cols)
RUN: PYTHONIOENCODING=utf-8 python build_gnn_relational_features.py
"""
import os
import numpy as np
import networkx as nx

GDIR = os.path.join("data", "yelp_graph")
COLS = ["rel_degree", "rel_comp_size", "rel_core", "rel_clustering",
        "rel_temporal_degree", "rel_reviewer_train_fraud", "rel_neighbor_fraud_rate"]


def main():
    g = np.load(os.path.join(GDIR, "graph_full.npz"))
    ru = g["review_user"]; rl = g["review_label"].astype(int); rs = g["review_split"].astype(int)
    Nu = g["reviewer_feat"].shape[0]; Nr = len(ru)
    e = np.load(os.path.join(GDIR, "reviewer_edges.npz"))["edge_index"]
    et = np.load(os.path.join(GDIR, "reviewer_edges_temporal.npz"))["edge_index"]
    print(f"reviewers {Nu:,} | reviews {Nr:,} | co-edges {e.shape[1]:,} | temporal {et.shape[1]:,}", flush=True)

    G = nx.Graph(); G.add_nodes_from(range(Nu)); G.add_edges_from(zip(e[0].tolist(), e[1].tolist()))
    Gt = nx.Graph(); Gt.add_nodes_from(range(Nu)); Gt.add_edges_from(zip(et[0].tolist(), et[1].tolist()))

    deg = np.array([d for _, d in sorted(G.degree())], dtype=np.float32)
    tdeg = np.array([d for _, d in sorted(Gt.degree())], dtype=np.float32)
    comp_size = np.ones(Nu, dtype=np.float32)
    for comp in nx.connected_components(G):
        s = len(comp)
        if s > 1:
            for n in comp: comp_size[n] = s
    core = nx.core_number(G); core_arr = np.array([core.get(i, 0) for i in range(Nu)], dtype=np.float32)
    clus = nx.clustering(G); clus_arr = np.array([clus.get(i, 0.0) for i in range(Nu)], dtype=np.float32)
    print("structural features done (degree/component/core/clustering/temporal)", flush=True)

    # label-aware, TRAIN-only
    tr = rs == 1
    cnt = np.zeros(Nu); fake = np.zeros(Nu)
    np.add.at(cnt, ru[tr], 1.0); np.add.at(fake, ru[tr], rl[tr].astype(float))
    rev_frac = np.where(cnt > 0, fake / np.maximum(cnt, 1), 0.0).astype(np.float32)
    nbr_fraud = np.zeros(Nu, dtype=np.float32)
    for i in range(Nu):
        nb = list(G.neighbors(i))
        if nb: nbr_fraud[i] = float(np.mean(rev_frac[nb]))
    print("label-aware (train-only) reviewer/neighbor fraud rates done", flush=True)

    # map per-reviewer -> per-review (aligned to features_full / GNN by row index)
    rel = np.column_stack([deg[ru], comp_size[ru], core_arr[ru], clus_arr[ru],
                           tdeg[ru], rev_frac[ru], nbr_fraud[ru]]).astype(np.float32)
    np.savez_compressed(os.path.join(GDIR, "relational_features.npz"), rel=rel, cols=np.array(COLS))
    # quick sanity: how discriminative is neighbor_fraud_rate (train) wrt label, on TRAIN?
    nz = (rel[:, 0] > 0)
    print(f"\nsaved relational_features.npz  rel {rel.shape}")
    print(f"reviews with any co-edge (degree>0): {nz.mean():.1%}")
    print(f"mean comp_size among coordinated: {comp_size[ru][nz].mean():.1f} | max {comp_size.max():.0f}")


if __name__ == "__main__":
    main()

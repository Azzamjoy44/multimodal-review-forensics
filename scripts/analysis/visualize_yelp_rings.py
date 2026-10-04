# --- repo-root bootstrap (added during reorg: keeps flat imports + data/ paths working) ---
import os as _os, sys as _sys, pathlib as _pl
_ROOT = next((p for p in _pl.Path(__file__).resolve().parents if (p / "main.py").exists()), _pl.Path.cwd())
_sys.path.insert(0, str(_ROOT))
for _d in ((_ROOT / "scripts").iterdir() if (_ROOT / "scripts").is_dir() else []):
    if _d.is_dir(): _sys.path.insert(0, str(_d))
_os.chdir(_ROOT)
# --- end repo-root bootstrap ---

"""
visualize_yelp_rings.py — draw the extracted coordinated fake-review rings.

Each ring: red nodes = fake reviewers, blue squares = the businesses they jointly
target, edges = fake reviews. A coordinated ring shows up as a dense cluster of
reviewers converging on the same businesses. Reads data/yelp_rings.csv (from
extract_yelp_rings.py) + the raw metadata for the reviewer->business edges.

RUN:  python visualize_yelp_rings.py   ->  yelp_rings_viz.png
"""

import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

import build_yelp_multimodal_features as M

MAX_SHARED_BIZ = 35   # cap businesses drawn per ring (most-shared first) for readability


def load_fake_edges():
    """user -> set of businesses they FAKE-reviewed (full raw data)."""
    ufb = defaultdict(set)
    def zipf(path, pre):
        for l in open(path, encoding="utf-8", errors="ignore"):
            m = l.rstrip("\n").split("\t")
            if len(m) >= 5 and m[3].strip() == "-1":   # -1 = fake (filtered)
                ufb[f"{pre}_{m[0].strip()}"].add(f"{pre}_{m[1].strip()}")
    def chif(path, pre):
        for l in open(path, encoding="utf-8", errors="ignore"):
            p = l.split()
            if len(p) >= 9 and p[4].strip().upper() == "Y":
                ufb[f"{pre}_{p[2]}"].add(f"{pre}_{p[3]}")
    zipf(M.YELPZIP_META, "zip"); zipf(M.YELPNYC_META, "nyc")
    chif(M.CHI_RES_META, "chi_res"); chif(M.CHI_HOT_META, "chi_hot")
    return ufb


def main():
    print("reading raw fake-review edges ...")
    ufb = load_fake_edges()

    rings = []
    for row in csv.DictReader(open("data/yelp_rings.csv", encoding="utf-8")):
        rings.append({"id": row["ring_id"], "n": int(row["n_reviewers"]),
                      "members": [u for u in row["reviewer_ids"].split("|") if u]})
    # pick four mid-size rings (skip the largest, which is too dense to read)
    pick = [r for r in rings if 5 <= r["n"] <= 16][:4]
    if len(pick) < 4:
        pick = sorted(rings, key=lambda r: abs(r["n"] - 10))[:4]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, r in zip(axes.ravel(), pick):
        members = [u for u in r["members"] if u in ufb]
        bizcount = defaultdict(int)
        for u in members:
            for b in ufb[u]:
                bizcount[b] += 1
        shared = [b for b, c in bizcount.items() if c >= 2]
        shared = sorted(shared, key=lambda b: bizcount[b], reverse=True)[:MAX_SHARED_BIZ]
        sset = set(shared)

        G = nx.Graph()
        G.add_nodes_from((("r", u) for u in members), bip="r")
        G.add_nodes_from((("b", b) for b in shared), bip="b")
        for u in members:
            for b in ufb[u] & sset:
                G.add_edge(("r", u), ("b", b))
        pos = nx.spring_layout(G, k=0.6, iterations=80, seed=3)
        rn = [n for n in G if n[0] == "r"]
        bn = [n for n in G if n[0] == "b"]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6)
        nx.draw_networkx_nodes(G, pos, nodelist=bn, ax=ax, node_color="#60a5fa",
                               node_shape="s", node_size=90)
        nx.draw_networkx_nodes(G, pos, nodelist=rn, ax=ax, node_color="#ef4444",
                               node_size=200, edgecolors="#7f1d1d", linewidths=0.8)
        ax.set_title(f"Ring {r['id']}:  {len(members)} fake reviewers  →  "
                     f"{len(shared)} shared target businesses", fontsize=12, weight="bold")
        ax.axis("off")

    import matplotlib.patches as mp
    fig.legend(handles=[mp.Patch(color="#ef4444", label="reviewer (100% fake)"),
                        mp.Patch(color="#60a5fa", label="business (shared target)")],
               loc="lower center", ncol=2, fontsize=11, frameon=False)
    fig.suptitle("Coordinated fake-review rings — reviewers jointly targeting the same businesses",
                 fontsize=16, weight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("yelp_rings_viz.png", dpi=130, bbox_inches="tight")
    print("saved yelp_rings_viz.png")


if __name__ == "__main__":
    main()

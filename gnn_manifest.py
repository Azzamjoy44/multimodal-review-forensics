"""
gnn_manifest.py — shared writer for the GNN-rings gallery manifest the frontend reads.

Manifest shape:  {"datasets": [ {"name": str, "levels": [ {level dict}, ... ]}, ... ]}
Each render_*_ring_levels.py calls upsert_dataset(name, levels) so multiple datasets
(Yelp, Amazon, ...) coexist in one manifest. Yelp is kept first.
"""

import os
import json

MANIFEST = os.path.join(os.path.dirname(__file__), "static", "gnn", "levels", "manifest.json")


def _load():
    if not os.path.exists(MANIFEST):
        return {"datasets": []}
    data = json.load(open(MANIFEST, encoding="utf-8"))
    if isinstance(data, list):                 # legacy flat (Yelp-only) list -> upgrade
        return {"datasets": [{"name": "Yelp", "levels": data}]}
    return data


def upsert_dataset(name, levels):
    data = _load()
    data["datasets"] = [d for d in data["datasets"] if d.get("name") != name]
    data["datasets"].append({"name": name, "levels": levels})
    data["datasets"].sort(key=lambda d: (d["name"] != "Yelp", d["name"]))   # Yelp first
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(data, open(MANIFEST, "w", encoding="utf-8"), indent=2)
    return MANIFEST

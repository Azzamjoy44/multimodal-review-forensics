# Deployment Guide — Real-Time Review Forensics

Deploy the full app to a **16 GB x86 (amd64) cloud VM**, CPU-only, via Docker. No GPU
is needed — serving is ONNX Runtime + sklearn inference.

## Sizing
| | |
|---|---|
| Runtime model files shipped | **~2.3 GB** (after the Random-Forest shrink — see `shrink_sentiment_rf.py` / `shrink_rfs.py`) |
| Resident RAM after all models load | **~3–4 GB** (models ~1.9 GB + Amazon review index ~1–2 GB + libs + inference buffers) |
| **VM recommendation** | **8 GB works comfortably; 16 GB for generous headroom**, 2–4 vCPU, **≥ 30 GB disk**, Ubuntu 22.04 LTS (amd64) |

Concrete picks: **AWS EC2 `t3.xlarge`**, **DigitalOcean 16 GB droplet**, **GCP `e2-standard-4`**.

---

## Step 1 — Provision the VM
Create an **Ubuntu 22.04 LTS, amd64, 8 GB RAM (16 GB for headroom), ≥30 GB disk** instance. Open inbound
ports **22** (SSH), **80** (HTTP) and **443** (HTTPS) in its firewall / security group.
SSH in: `ssh ubuntu@<VM_IP>`.

## Step 2 — Install Docker (on the VM)
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker   # run docker without sudo
```

## Step 3 — Copy the project to the VM (slim set only)
The models are on your machine. Transfer the project **excluding the ~6 GB of training
data + replaced models** (same set as `.dockerignore`). From your machine
(use Git Bash / WSL on Windows, or WinSCP for the GUI route):

```bash
rsync -avz --progress \
  --exclude '.git' --exclude '__pycache__' --exclude '*.ipynb' \
  --exclude 'data/merged_sentiment.csv' --exclude 'data/tweets.csv' \
  --exclude 'data/IMDB Dataset.csv' --exclude 'data/fake reviews dataset.csv' \
  --exclude 'data/yelp_split*.csv' --exclude 'data/yelp_multimodal_features.csv' \
  --exclude 'data/yelp_multimodal_augmented.csv' --exclude 'data/review_detection.csv' \
  --exclude 'data/non_reviews.csv' --exclude 'data/yelp_frontend_reviews.csv' \
  --exclude 'data/ai_*.csv' --exclude 'data/*_results.csv' \
  --exclude 'data/fake_reviews_broadened.csv' \
  --exclude 'data/YelpZip*' --exclude 'data/YelpNYC*' --exclude 'data/YelpChi*' \
  --exclude 'data/fake_*.joblib' --exclude 'data/fake_lstm.onnx' \
  --exclude 'data/fake_lstm_tokenizer.json' --exclude 'data/fake_distilbert_onnx' \
  --exclude 'data/yelp_fake_*' \
  --exclude 'data/yelp_multimodal_fusion_models' \
  --exclude 'data/yelp_multimodal_generalized_models' \
  --exclude 'data/users.db' \
  ./ ubuntu@<VM_IP>:/home/ubuntu/review-forensics/
```
This transfers ~2.3 GB. (If `rsync` isn't available on Windows, WinSCP works — just
manually skip the files/folders listed above.)

## Step 4 — Build the image (on the VM)
```bash
cd ~/review-forensics
docker build -t review-forensics .
```
Takes a few minutes (CPU torch + deps + copying ~2.3 GB of models). Final image ~4–5 GB.

## Step 5 — Run it
```bash
docker run -d --name rf -p 80:8000 --restart unless-stopped review-forensics
```
- `-p 80:8000` serves the app on port 80 (the app listens on 8000 inside).
- `--restart unless-stopped` survives reboots/crashes.

## Step 6 — Verify
```bash
docker logs -f rf      # wait for "All models loaded — server ready." (~30–60 s)
```
Then open `http://<VM_IP>/` in a browser, or `http://<VM_IP>/docs` for the API.

## Step 7 (optional) — Domain + HTTPS
Point a domain's A-record at `<VM_IP>`, then put **Caddy** in front for automatic TLS.
`docker-compose.yml`:
```yaml
services:
  app:
    build: .
    restart: unless-stopped
    expose: ["8000"]
  caddy:
    image: caddy:2
    restart: unless-stopped
    ports: ["80:80", "443:443"]
    command: caddy reverse-proxy --from your-domain.com --to app:8000
    volumes: ["caddy_data:/data"]
volumes: { caddy_data: {} }
```
`docker compose up -d --build` → HTTPS at `https://your-domain.com` with auto-renewing certs.

---

## Reflecting edits

### A. Live-edit mode (recommended while developing) — edit one line, see it immediately
Edit files **directly on the VM** and let the server pick them up automatically. No
sync, no rebuild.

1. **Edit on the VM with VS Code Remote-SSH:** in VS Code install the *Remote - SSH*
   extension → "Connect to Host" → `ubuntu@<VM_IP>` → open `/home/ubuntu/review-forensics`.
   Now your editor reads/writes the VM's files directly — it feels exactly like local.
2. **Run dev mode** (live-mounts the code + runs `uvicorn --reload`):
   ```bash
   docker compose -f docker-compose.dev.yml up -d --build   # first time
   ```
3. **Edit and see it:**
   - **`static/index.html` / frontend** → save, then **refresh the browser. Instant.**
     (No restart — static files are served live through the mount.)
   - **A `.py` file** → save; uvicorn auto-restarts and the app is back after the models
     reload (**~30–60 s** — unavoidable, since the ~8–10 GB of models reload into the new
     process). Still zero manual steps.
   - **A retrained model file** → replace it, then
     `docker compose -f docker-compose.dev.yml restart app`.

That's the "edit one line → reflected" loop. Use the stable build (below) when you're
done iterating.

### B. Stable / production update (baked image)
The plain `docker-compose.yml` bakes everything into the image (more robust, no
`--reload`). To push a change: **sync changed files to the VM → rebuild → recreate**,
collapsed into one command.

**1. Sync your changes to the VM** (same `rsync` as Step 3 — it only sends changed files):
```bash
rsync -avz --progress <same --exclude flags as Step 3> ./ ubuntu@<VM_IP>:/home/ubuntu/review-forensics/
```

**2. Rebuild + restart** (on the VM):
```bash
cd ~/review-forensics
docker compose up -d --build      # rebuilds only changed layers, recreates the container
docker image prune -f             # optional: drop the old image layers
```

Cost by change type:
- **Python / frontend code** — `up -d --build` rebuilds in seconds (the pip layer is
  cached; only the small code copy re-runs).
- **`requirements.txt`** — the pip layer re-runs (a few minutes).
- **Retrained model files** — re-sync just the changed `data/<model_dir>/`, then
  `up -d --build` re-copies them into the image and restarts.

**Faster iteration (skip rebuilds):** uncomment the `volumes:` block in
`docker-compose.yml` to mount `data/` and `static/` from the host. Then a model retrain
or frontend tweak needs only a re-sync + `docker compose restart app` (frontend edits
show on a plain browser refresh) — no image rebuild at all.

**Recommended hygiene — put the *code* in Git.** Track the `.py` / `static/` / config
in a Git repo (push from your machine, `git pull` on the VM) for history + easy rollback.
Keep the big model files **out of Git** (use this `rsync` flow, Git LFS, or object
storage) — they're too large for a normal repo. This project isn't a Git repo yet; I can
`git init` it with a sensible `.gitignore` (excluding `data/` + the training files) if
you want that workflow.

---

## Operational notes
- **Single worker only.** The Dockerfile runs `--workers 1` on purpose: each worker
  loads the full ~8–10 GB model set, so multiple workers would multiply RAM. One worker
  on a 16 GB box is correct. (For more concurrency, scale to a bigger VM and add workers,
  or run multiple containers behind Caddy.)
- **Startup** pre-loads every model (~30–60 s) before the first request — normal.
- **Auth** (`data/users.db`) is created on first run; in-memory sessions reset whenever
  the container restarts (by design).
- **Updating models/code:** re-run Step 3 (rsync) then `docker compose up -d --build`
  (or rebuild + `docker restart rf`).

### Footnote: Random-Forest shrink (already applied)
The three Random Forests (sentiment, review-detector, Model B) were originally trained
with unbounded tree depth → 2.19 GB / 797 MB / 531 MB. They've since been retrained
depth-bounded (`shrink_sentiment_rf.py`, `shrink_rfs.py`) to **17 MB / 114 MB / 55 MB** —
~3.3 GB reclaimed for ≤2.4 pp macro-F1 on those individual ensemble members (immaterial
to the consensus verdicts). The training scripts are bounded too, so future retrains
stay small.

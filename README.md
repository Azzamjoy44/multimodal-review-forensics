# A Multi-Modal Adversarial Network for Real-Time Review Forensics

Bachelor's thesis web app for **fake-review detection**, **sentiment analysis**, and a
**review/non-review gate** across Amazon and Yelp. FastAPI backend + a single-file
vanilla-JS frontend. All ML models are trained offline and pre-loaded at startup.

The live app runs one generalized fake detector per domain:
- **Unified text detector** — CG + human-deception detector → Amazon Fake Review Detection section + Dashboard *Analyze* box.
- **Model B** — multi-modal (text + behavioural) detector → Yelp Fake Review Detection section.

## Run it locally
```powershell
uvicorn main:app --reload      # http://localhost:8000   (API docs: /docs)
```
Needs **~4–5 GB RAM free** — ~1.9 GB of models load at startup plus the in-memory review
indexes (~1–2 GB). On a memory-constrained machine, host it instead (see below).

## Edit with changes reflected immediately (on a server)
1. Open the VM in **VS Code Remote-SSH** (edit files directly on the server).
2. ```bash
   docker compose -f docker-compose.dev.yml up -d --build
   ```
   - **Frontend** (`static/index.html`) edits → refresh the browser. **Instant.**
   - **Python** edits → server auto-reloads (~30–60 s while the models reload).

## Deploy (stable)
```bash
docker compose up -d --build
```
Full hosting guide — provision a 16 GB VM, ship only the runtime files, HTTPS, and the
update loop — is in **[DEPLOY.md](DEPLOY.md)**.

## Data & models

The datasets and trained models are **not** in this repository (~12 GB, and
several sources are not redistributable). See **[DATA.md](DATA.md)** for the
download archive, the per-dataset sources and licences, and the scripts that
rebuild the derived data. In short: download the model archive into `data/`, and
obtain the restricted raw datasets (notably Yelp) from their original sources.

## Docs
| File | What's in it |
|---|---|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Request flow, startup pre-loading, scoring pipeline, model map |
| **[DATA.md](DATA.md)** | Datasets & models: sources, licences, how to obtain and rebuild |
| **[DEPLOY.md](DEPLOY.md)** | Hosting on a 16 GB VM, dev vs stable, updating the deployment |
| **[THESIS_NOTES.md](THESIS_NOTES.md)** | Model designs, training recipes, all evaluation results |

## Key files
| | |
|---|---|
| `main.py` | FastAPI app, endpoints, startup pre-loading |
| `score_reviews.py` | Central scoring (rules + sklearn + LSTM/DistilBERT + highlight phrases) |
| `model_a.py` / `model_b.py` | Generalized Amazon detector / multi-modal Yelp detector |
| `review_detector.py` | Review vs non-review gate for the Analyze box |
| `load_amazon_reviews.py` / `load_yelp_reviews.py` | In-memory review indexes |
| `static/index.html` | Entire frontend (no build step) |
| `Dockerfile`, `docker-compose.yml`, `docker-compose.dev.yml`, `.dockerignore` | Containerized run / dev / deploy |

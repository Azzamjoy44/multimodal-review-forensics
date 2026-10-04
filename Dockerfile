# Real-Time Review Forensics — serving image (CPU-only inference).
# Build:  docker build -t review-forensics .
# Run:    docker run -d --name rf -p 80:8000 --restart unless-stopped review-forensics

FROM python:3.12-slim

# NOTE: image OCR is now Mistral (a network API) — no local OpenCV/torch needed for it.
# The libs + torch below are LEGACY (were for EasyOCR) and are safe to PRUNE on the next
# Docker rebuild+test: nothing in the live serving path imports torch (review-detection and
# sentiment run on onnxruntime; torch is only the offline BERT-sentiment eval path). Kept
# for now only to avoid an unverified build change.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Legacy CPU-only torch (see note above — prunable). CPU index avoids the ~2.5 GB CUDA build.
RUN pip install --no-cache-dir torch==2.12.0 torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the slim runtime data set (.dockerignore drops training data + replaced models)
COPY . .

EXPOSE 8000

# IMPORTANT: a single worker. The models are sizable (~1.9 GB on disk, ~3-4 GB resident);
# multiple uvicorn workers would each load the full model set and multiply RAM use.
# All models pre-load in the FastAPI lifespan at startup (~30-60 s before first request).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

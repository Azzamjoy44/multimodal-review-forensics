# Data and models

The code in this repository is small; the datasets and trained models are large
(~12 GB raw) and are **not** stored in git. This file explains where everything
comes from, what is redistributable, and how to rebuild the derived artifacts.

> **Archive (trained models + shareable derived data):**
> `<ZENODO DOI / LINK — add after upload>`
>
> Download the archive and unzip it into `data/` at the repository root, then
> run the app (`uvicorn main:app --reload`). For anything that cannot be
> redistributed (see below), use the listed source + build script to regenerate it.

---

## 1. What is shareable vs. what you must obtain yourself

**Shared in the archive (ours to distribute):**
- **Trained model weights** — the sklearn (`*.joblib`), BiLSTM/DistilBERT (`*.onnx`)
  and GNN artifacts. These are our own trained outputs and are the fastest way to
  reproduce the system and the paper's numbers.
- **Data we generated** — e.g. the AI-written reviews used for augmentation.
- Small derived CSVs whose source licences permit redistribution.

**You must obtain yourself (third-party, not redistributable here):**
- The **Yelp** datasets (YelpZip / YelpNYC / YelpChi) are released by their authors
  on request and **may not be redistributed**. Request them from the source below.
- Any derived file built *from* restricted data inherits the restriction — in
  particular `yelp_split.csv`, `yelp_multimodal_features*.csv`, and
  `yelp_frontend_reviews.csv` (which contains raw Yelp review text). These are
  **not** in the archive; rebuild them with the scripts below after you have the raw data.
- The large public corpora (Amazon, IMDB, Sentiment140, CG/OR, HC3) are easiest to
  pull from their original sources under their own terms.

---

## 2. Raw datasets — sources

| Dataset | Used for | Source | Notes / terms |
|---|---|---|---|
| Amazon Fine Food Reviews (`Reviews.csv`) | live Amazon data + sentiment/false-positive eval | McAuley & Leskovec, WWW 2013 — Kaggle / SNAP | redistribution discouraged; download from source |
| Fake reviews dataset (CG/OR) | unified text detector (machine-vs-human) | Salminen et al., *J. Retail. Consum. Serv.* 2022 — Kaggle / OSF | see source licence |
| IMDB movie reviews | sentiment training | Maas et al., ACL 2011 — ai.stanford.edu / Kaggle | academic use |
| Sentiment140 (`tweets.csv`) | sentiment training | Go, Bhayani & Huang, 2009 — Stanford | academic use |
| YelpZip / YelpNYC / YelpChi | Yelp fake detection, Model B, GNN | Rayana & Akoglu, KDD 2015 (SpEagle) — **request from the authors** | **NOT redistributable** |
| HC3 | out-of-distribution probe | Guo et al., arXiv 2023 — Hugging Face `Hello-SimpleAI/HC3` | see source |
| Amazon Musical Instruments fraud graph | Amazon GNN benchmark | CARE-GNN (Dou et al., CIKM 2020) / PC-GNN (Liu et al., WWW 2021) | from the authors' repos |
| Review-gate auxiliary corpora (Steam, app reviews, SQuAD, news, wiki, CodeSearchNet, LexGLUE, …) | review/non-review gate | mostly Hugging Face / public; auto-fetched by `download_negatives.py` | per-source terms; full list in THESIS_NOTES.md |

The complete per-source list with citations is in **THESIS_NOTES.md → "Review / Non-Review Detection — Dataset Sources"** and in the paper's reference list.

---

## 3. Derived datasets — rebuild with these scripts

| Output (in `data/`) | Built by | Inputs |
|---|---|---|
| `merged_sentiment.csv` | `merge_sentiment_datasets.py` | IMDB + Sentiment140 |
| `non_reviews.csv` | `download_negatives.py` | ~18 public non-review sources |
| `review_detection.csv` | `build_review_dataset.py` | reviews + `non_reviews.csv` |
| `yelp_split.csv` *(restricted)* | `prepare_yelp_split.ipynb` | Yelp raw (grouped, leakage-free split) |
| `yelp_frontend_reviews.csv` *(restricted)* | `prepare_yelp_frontend.py` | Yelp raw (test split + metadata) |
| `yelp_multimodal_features*.csv` *(restricted)* | `build_yelp_multimodal_features.py` | Yelp raw (17 behavioural features) |
| combined CG + human-deception set | `build_broadened_cgor.py` | CG/OR + Ott/Li corpora + open LLM reviews |
| AI-augmented Yelp set | `build_yelp_multimodal_augmented.py` | Yelp features + generated AI reviews |

Deep models (BiLSTM / DistilBERT / GNN) are trained in the Colab notebooks
(`train_*.ipynb`); classical models and all evaluation run locally. See
**THESIS_NOTES.md** for the exact recipes and **ARCHITECTURE.md** for the model map.

---

## 4. Trained models (in the archive)

The served models load from `data/` at startup (see `main.py`). The key ones:
`combined_fake_*` (unified text detector), `yelp_multimodal_full_*` (Model B),
`review_detector_*` (review gate), `sentiment_*` / `distilbert_sentiment_onnx`
(sentiment), and the GNN ring artifacts. Model-weight files are our own outputs
and are redistributable; they do not contain raw third-party review text.

---

## 5. If you just want to run the app

1. Obtain the Yelp raw data from the source above (or skip the Yelp section).
2. Download the model archive from the Zenodo link and unzip into `data/`.
3. `pip install -r requirements.txt`
4. `uvicorn main:app --reload` → http://localhost:8000

For image OCR in the Analyze box, set `MISTRAL_API_KEY` in your environment.

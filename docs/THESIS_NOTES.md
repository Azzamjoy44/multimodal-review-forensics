# Thesis Notes — Useful Findings

This file records findings, statistics, and observations that may be useful
for writing or presenting the thesis. Updated as new information is found.

---

## Datasets

### Amazon Fine Food Reviews
- 568,000 reviews
- Used as the live data source for the Amazon pipeline in the frontend
- No fake/genuine labels — used for live scoring and dashboard only

### CG/OR Fake Reviews Dataset (Kaggle)
- 40,432 reviews — balanced: ~20,216 fake (CG) + ~20,216 genuine (OR)
- Fake reviews are **computer-generated** text
- Used to train all Amazon fake detection models (sklearn × 8, LSTM, DistilBERT)

### YelpZip
- 608,598 reviews — 80,466 fake (13.2%) + 528,132 genuine (86.8%)
- Labels derived from Yelp's spam filter (behavioral signals, not text patterns)

### YelpNYC
- 359,052 reviews — 36,885 fake (10.3%) + 322,167 genuine (89.7%)
- Included in training (same Yelp platform, same labeling mechanism as YelpZip/YelpChi)
- Test-split reviews (cross-referenced from raw files) contribute to the frontend dataset

### YelpChi
- Restaurant subset: 61,541 reviews — 8,141 fake (13.2%) + 53,400 genuine
- Hotel subset: 5,854 reviews — 778 fake (13.3%) + 5,076 genuine
- Combined: 67,395 reviews at ~13.3% fake
- Labels derived from Yelp's spam filter

### Merged Sentiment Corpus
- ~1.65 million rows — IMDB (50k) + Sentiment140 tweets (1.6M)
- Used to train all sentiment analysis models
- 500,000 row sample used for sklearn models (training time constraint)

---

## Generalization & Out-of-Distribution (OOD) Evaluation

A fake-review detector is only credible if it works on data it never trained on. Beyond
the standard held-out test splits (same distribution), the two detectors are evaluated
on three harder, increasingly out-of-distribution axes: **unseen generators**,
**unseen domains**, and an **external corpus** (different source *and* format). Detailed
tables live in the per-model sections below; this section consolidates the results.

### Model A — all-domain AI-review detector (CG/OR)

**1. External zero-shot — HC3** (`evaluate_cgor_external.py`; 8,000 balanced
Human-vs-ChatGPT texts — a different corpus *and* format: Q&A answers, not reviews):

| model | HC3 macro-F1 | ChatGPT recall | human spec |
|---|---|---|---|
| **DistilBERT** | **90.1%** | 87.8% | 92.4% |
| sklearn (8 models) | 42–48% | 12–26% | 77–94% |
| LSTM | 40.1% | 7.2% | 97.4% |
| naïve 10-model majority | 45.0% | 13.2% | 95.3% |

*Key finding:* **DistilBERT transfers (90% zero-shot on an unseen domain+format); the
TF-IDF sklearn models and GloVe-LSTM do not (~40–48%).** The bag-of-words/static-
embedding models learned *review-surface vocabulary* that vanishes in Q&A text and
collapse to "predict human"; DistilBERT learned a deeper, transferable machine-text
representation. → On OOD input, trust the transformer, not an equal-weight majority.

**1b. Same HC3 probe on the SERVED detector (`model_unified`, CG/OR + human deception) — the deployed-model OOD number** (`evaluate_unified_hc3_colab.ipynb` GPU DL → `unified_hc3_dl_preds.csv`; `merge_unified_hc3.py` local sklearn; same 8,000 balanced HC3; serves the Analyze box + the frontend Amazon "Out-of-Distribution" toggle):

| model | HC3 macro-F1 | ChatGPT recall | human spec |
|---|---|---|---|
| **DistilBERT** | **77.0%** | 65.9% | 88.8% |
| MLP (best sklearn) | 52.0% | 24.6% | 89.4% |
| LSTM | 48.5% | 18.2% | 93.5% |
| other sklearn | 38–48% | 7–24% | 79–94% |
| 8-sklearn majority | 37.0% | 4.0% | 97.2% |
| **full 10-model ensemble** | **38.7%** | 5.6% | 97.7% |

Same qualitative story, **lower transfer than the broadened detector**: the served unified DistilBERT generalizes to HC3 at **77.0% macro-F1 / 65.9% ChatGPT recall** (vs ~95–100% in-distribution → a ~20 pp OOD drop), while sklearn/LSTM and the equal-weight ensemble **collapse to ~38%** (the 8 non-transferring sklearn outvote the 2 deep members → the 10-model majority catches only **5.6%** of ChatGPT text, ~38 pp *below* its own best single model). **The served detector's defensible OOD number is DistilBERT's 77%, not 90%** (90% was the older broadened model; `model_unified` is the superset trained on CG/OR + human deception and transfers somewhat less). Confirms the deployment rule: **on OOD, trust the transformer, never the majority vote.** *(The HC3 result stays a reported thesis number; the live frontend "Out-of-Distribution (HC3)" toggle that previously demoed it was **removed** from the Amazon section — the Dataset dropdown now offers only the labeled held-out test set and the Amazon Fine Food browse. The backend still supports `dataset='ood'` on `/amazon-fake-labeled` for offline reproduction; it's just no longer surfaced in the UI.)*

**2. Leave-one-generator-out** (`evaluate_cgor_generalization.py`; each generator fully
held out): **mean 0.79** across 14 families — modern LLMs strong (gpt 1.00, llama3 0.97,
zephyr 0.94, yi 0.93, qwen 0.91, openai_gpt 0.89, sutro 0.87, lyra 0.83), old/weak
moderate (distilgpt2 0.76, gpt2 0.73, opt350 0.66, gpt2med 0.64), **salminen_cg 0.15** (floor).

**3. Leave-one-domain-out** (`evaluate_cgor_leave_one_domain.py`; each domain held out):
**mean 0.77** — movies 0.97, books 0.93, games 0.87, apps 0.83, restaurants 0.70, **products 0.33**.

### Model B — multi-modal Yelp detector (text + behavioral)

**Leave-one-generator-out (AI)** (`evaluate_yelp_ai_logo.py`; each AI generator fully
held out): **mean unseen-generator recall 86.9%** (zephyr 91.3%, gpt 89.2%, yi 87.2%,
llama3 83.9%, qwen 82.9%) — only ~6 pp below in-distribution (92.5%), i.e. it learned
general "machine-ness", not per-generator fingerprints.

**Held-out clean Yelp test vs the text-only baseline** (`evaluate_yelp_modelb_vs_baseline.py`;
identical reviews, no AI fakes, so "fake recall" = human-fake recall): the deployed Model B
(early-fusion, trained on the AI-augmented objective with ×10 AI weight) scores **below**
text-only on the pure human-fake task — but that comparison is confounded, because Model B
was optimized for a *different* objective (catch AI **and** human fakes), not pure Yelp.

**Does a PROPERLY-built fusion beat text-only? The fair test (`make_modelb_good.py`).** Every
prior fusion was handicapped: the served Model B trained for the AI-augmented objective, and
the "feature-level fusion" (`train_yelp_multimodal_fusion.py`) used a single weak LR text branch
(~67%) and only ever compared fusion to *that same weak LR*. Nobody fused behavioral onto the
**strong** text signal. So: stack the full text-only ensemble's 10 per-model probabilities
(8 sklearn + LSTM + DistilBERT) and learn a meta-classifier in two forms — **text-stack** (10 text
probs) vs **full-stack** (10 text probs + 17 behavioral) — leakage-controlled (text models trained
on TRAIN, meta-clf trained on VAL, evaluated on TEST). Result (clean Yelp test, n=3,000):

| model | macro-F1 | fakeR | fakeP |
|---|---|---|---|
| text: DistilBERT only | 70.0% | 72.9% | 68.8% |
| behavioral-only (17 feat) | 73.7% | 82.9% | 70.1% |
| text-stack (10 probs) | 90.7% | 86.2% | 94.6% |
| **full-stack (10 probs + 17 beh)** | **90.9%** | 88.9% | 92.6% |

**Two findings — one resolving the user's intuition, one a clean negative result:**
1. **A properly-built fusion is NOT worse than text-only** (full-stack 90.9% ≥ text-stack 90.7%). So the deployed Model B's underperformance really *was* a build defect (early fusion + the AI-augmented ×10 objective), exactly as suspected — not a fundamental limit.
2. **But behavioral adds only +0.2 pp on top of strong, well-combined text.** Even given every fair chance (strong text branch, leakage-controlled stacking), the 17 per-review behavioral features are **redundant** with what the text models already capture — they don't meaningfully add. On Yelp, fusion-as-features has no upside.

**⚠️ The absolute ~90% is inflated and is NOT a deployable accuracy.** The best individual text model here is ~81% (rf 80.1 / mlp 80.7 / dt 79.3); a 10-pp stack gain over the best member is implausible for honest stacking. The likely cause is **val↔test reviewer-level pattern sharing**: `yelp_split` is grouped by *text*, not *reviewer*, so the same reviewers span val and test, the tree models memorize each reviewer's templated phrasing, and the meta-clf (trained on val) exploits those per-reviewer probability patterns on test. The **paired delta (+0.2 pp)** is internally valid regardless (both stacks share identical text probs + protocol); only the absolute level is suspect. A reviewer-grouped split would be needed for a clean absolute number, but it wouldn't change the redundancy conclusion.

**GNN ⊕ Model B fusion — tested, doesn't help (honest negative result).** Asked whether fusing the relational GNN with per-review Model B beats either alone (`experiment_gnn_modelb_fusion.py`, `build_gnn_relational_features.py`, `experiment_gnn_relational_fusion.py`; GNN temporal preds align row-for-row with `features_full`, leakage-controlled val→test). On the clean test: Model B 76.6–77.6%, GNN 76.2%, **score-level fusion 76.9% (+0.3), feature-level fusion (6 leakage-safe graph features: degree/k-core/clustering/temporal-degree/neighbor-fraud-rate) 77.8% (+0.2), +GNN-stack 77.5% (−0.1)**. The two models **agree ~91%** (71.9% both right, 18.9% both wrong); the **oracle ceiling is only 81.1% accuracy** (best a perfect per-review referee could do) and **no fusion approached it**. *Why:* (a) the graph features are **redundant with Model B's behavioral aggregates** (user burst/count, biz stats already encode "busy cluster"); (b) the GNN's complementary ~4.5% comes from **deep multi-hop propagation that doesn't reduce to per-reviewer scalars or to its final score**; (c) the disagreements are symmetric, so there's no routable signal. **Conclusion: per-review (Model B) and relational (GNN) signals are largely redundant on Yelp; keep them as two views — Model B for per-review detection, GNN for coordinated-ring *visualization*.**

**Unifying takeaway:** the *additive* "behavioral" signal on Yelp is **relational** (coordinated rings — the GNN), **not** per-review aggregate features. Behavioral-as-features ≈ redundant with text; behavioral-as-graph-structure (GNN) is the genuine orthogonal modality. So the honest multi-modal story is: **text-only** for per-review Yelp detection + **GNN** for coordinated rings + **the unified detector** for AI/cross-domain — *not* per-review text⊕behavioral fusion.

**Full-data Model B (pure, no AI) — trained on the entire deduped 678k corpus.** The served Model B was confounded (AI-augmented ×10 objective). Two clean retrains of the multi-modal sklearn ensemble (ColumnTransformer: TF-IDF text + MinMax 17 behavioral, threshold-tuned on val), no AI, on the **same Yelp test** as every other model: `train_yelp_multimodal_pure_sklearn.py` on the **178k balanced** subset → ENSEMBLE **75.1%** / xgb 75.7%; then on the **full 642k-train deduped** data (`yelp_multimodal_features_full.csv`, → `data/yelp_multimodal_full_sklearn_models/`) → **xgb 77.3%** (fakeR 82.4 / fakeP 74.7), ENSEMBLE **76.2%**, most models 75–77%. **Full vs 178k: xgb +1.6 pp, ensemble +1.1 pp** — training on all the behavioral history (not the balanced subsample) gives a modest real gain. Both clear text-only (69%) and edge past behavioral-only (76%). This is the honest, deployable Model B number: **~76–77% macro-F1**.

**Deep pure-full Model B (LSTM, end-to-end fusion).** `train_yelp_multimodal_lstm_full.ipynb` (GloVe+BiLSTM text branch ⊕ MLP behavioral branch, trained on the same full 642k pure data; imbalance via `class_weight='balanced'` {0:0.56, 1:4.51} + val-tuned threshold; early stop restored epoch 8) → **test macro-F1 77.1%** (fake recall 81.1% / precision 75.0% / genuine spec 73.2%, thr 0.5). **Matches the best sklearn (xgb 77.3%)** and beats text-only by ~8 pp — confirming **~77% is the robust multi-modal ceiling across architectures** (sklearn ≈ LSTM), not a single-model artifact. Training was clean (val 77.3% ≈ test 77.1%, mild post-peak overfit caught by early stopping); the balanced `class_weight` left the optimal threshold at a neutral 0.5 (calibrated, not threshold-rescued). Artifacts → `data/yelp_multimodal_lstm_full_onnx/` (feature order == `model_b.FEAT_COLS`, serving-compatible).

**Deep pure-full Model B (DistilBERT, end-to-end fusion).** `train_yelp_multimodal_distilbert_full.ipynb` (DistilBERT [CLS] text branch ⊕ MLP behavioral branch; imbalance via BCE `pos_weight = n_neg/n_pos ≈ 8` + val-tuned threshold; LLRD fine-tune, early stop restored epoch 2) → **test macro-F1 77.27%** (fake recall 82.4% / precision 74.7% / genuine spec 72.3%, thr 0.5) — the **best single Model B member**, marginally above xgb (77.3%) and LSTM (77.1%). Fast convergence (val peak epoch 2, overfit after — caught by early stopping; val 77.2% ≈ test 77.3%). Artifacts → `data/yelp_multimodal_distilbert_full_onnx/` (feature order == `model_b.FEAT_COLS`, serving-compatible).

**Pure-full Model B — all families converge on ~77%:** DistilBERT 77.27% · xgb 77.3% · LSTM 77.1% · sklearn-ensemble 76.2%. Three different architectures (boosted trees / BiLSTM / transformer) within **0.2 pp** of each other → **~77% is the robust multi-modal ceiling** on the behavioral-label Yelp task (not a model artifact), ~8 pp over text-only (69%). This is the clean, no-AI, full-data Model B — ready to replace the confounded AI-augmented served models across all three families.

**Exact per-member stats on the SERVED frontend test (`model_b_member_stats.py`, n=17,891 balanced 50/50; 8 multimodal sklearn scored live + LSTM/DistilBERT from `dl_predictions_all.csv`):**

| member | macro-F1 | fake recall | fake precision | genuine spec |
|---|--:|--:|--:|--:|
| DistilBERT | 77.3% | 82.4% | 74.8% | 72.3% |
| XGBoost | 77.0% | 82.2% | 74.5% | 72.0% |
| LSTM | 77.0% | 80.6% | 75.2% | 73.5% |
| Logistic Regression | 75.7% | 80.1% | 73.6% | 71.4% |
| AdaBoost | 75.7% | 85.0% | 71.8% | 66.8% |
| Random Forest | 75.6% | 80.5% | 73.3% | 70.8% |
| MLP | 75.2% | 79.9% | 73.0% | 70.6% |
| Naïve Bayes | 72.5% | 72.5% | 72.4% | 72.5% |
| KNN | 66.3% | 58.0% | 69.8% | 75.0% |
| Decision Tree | 58.5% | 34.1% | 75.7% | 89.1% |
| **8-sklearn ensemble** | 76.3% | 76.8% | 75.9% | 75.8% |
| **full 10-model served ensemble** | **76.7%** | 78.6% | 75.6% | 74.8% |

Top tier (77%): DistilBERT/XGBoost/LSTM. Weak members DT (58.5%, over-conservative: fakeR 34.1% but genSpec 89.1%) and KNN (66.3%) drag the vote, so the 10-model ensemble (76.7%) doesn't beat the best single member — consistent with the unified-detector finding. Members trade recall vs specificity (AdaBoost highest fakeR 85.0% / lowest genSpec 66.8%; DT the opposite). The full ensemble 76.7% matches the 4-detector frontend table exactly.

**Data-integrity fix — chi reviewer column in the SERVING features.** A thorough audit (`verify_everything_ok.py`) caught a real bug: `build_yelp_multimodal_features.py` (which feeds the served `yelp_frontend_multimodal.csv`) keyed YelpChi reviewers by `parts[1]` (= review_id) instead of `parts[2]` (= reviewer_id), so **every Chicago reviewer looked like a singleton** (all `user_*` aggregates wrong; `biz_*` and all Zip/NYC features were fine). The pure-full models trained on the *fixed* builder (`build_yelp_multimodal_features_full.py`, `parts[2]`), so serving them on the old CSV would feed wrong chi reviewer features on ~10% of rows. Fixed `parts[1]→parts[2]`, regenerated `yelp_multimodal_features.csv` + `yelp_frontend_multimodal.csv`; re-verified: **served features now byte-identical to the full-training features (max |delta| 0), all integrity checks pass.** Model-quality numbers above are unaffected (they were trained+tested on the fixed full file throughout).

**Corrected frontend comparison — Model B vs text-only on the served 17,891 (`compare_dl_colab.ipynb` GPU DL + `compare_merge_sklearn.py` local sklearn; scorer pre-validated 100% vs canonical paths).** The **NEW pure-full Model B wins decisively. Honest ensemble macro-F1 76.7% vs honest text-only 68.6% (+8.1 pp); leakage-free DL-only confirms 76.9% vs 68.5% (+8.4 pp).**

**IMPORTANT — earlier "+1.6 pp" was a contamination artifact, now corrected (`verify_baseline_contamination.py`).** The *served* baseline sklearn (`data/yelp_fake_sklearn_models/`) were trained on an **older `yelp_split`** (its metrics CSV shows `n_test=25,317`; the current split's test is 17,891). When `yelp_split` was regenerated, some of those models' old TRAINING reviews landed in the current frontend TEST set → **train/test contamination**. The high-capacity models memorize their training data, so on the frontend they inflate enormously: **dt +21.2 pp (served 78.1 → fresh 56.9), rf +14.7 (80.1 → 65.4), mlp +12.3 (80.2 → 67.9)**; linear/NB/xgb/adaboost barely move (+1.5–3.3); DL (LSTM/DistilBERT, don't memorize) show the honest ~69% throughout. That inflation propped the *served* text-only ensemble to 75.4% — the illusion that text-only was competitive. **Model B is verified NOT contaminated (0.00% train∩frontend-test overlap), so its ~77% is honest.** A freshly-retrained baseline on the *current* split drops to 68.6%, confirming Model B's true ~+8 pp lead. **Fix: retrain the served baseline sklearn on the current `yelp_split`** (the DL baselines are fine — DL doesn't memorize, so no contamination). **Decision stands: serve the pure-full Model B** (`yelp_multimodal_full_sklearn_models/` + `yelp_multimodal_lstm_full_onnx/` + `yelp_multimodal_distilbert_full_onnx/`). *(Baseline sklearn retrained on the current split — `retrain_yelp_baseline_sklearn.py` + `finish_baseline_mlp.py`; honest sklearn-majority 67.0%, vs the contaminated ~75%. Contaminated originals backed up to `yelp_fake_sklearn_models_stale_backup/`.)*

**CLEAN current-split 4-detector comparison (thesis table).** All four Yelp detectors on the SAME corrected frontend test (n=17,891, 10-model ensemble as served; DL re-scored on GPU via `compare_dl_all_colab.ipynb` → `dl_predictions_all.csv`, sklearn merged locally via `merge_all_detectors.py`):

| detector | ensemble macro-F1 | fakeR / fakeP | LSTM | DistilBERT |
|---|---|---|---|---|
| **Model B** (text+behavioral) | **76.7%** | 79% / 76% | 77.0% | 77.3% |
| baseline (text-only) | 68.4% | 70% / 68% | 68.4% | 69.6% |
| focal | 67.8% | 65% / 69% | 64.7% | 65.5% |
| contrastive | 67.9% | 65% / 69% | 64.5% | 65.3% |

**Findings:** (1) **Model B wins by +8.3 pp** — the behavioral modality is the lift; all text-only variants plateau at the ~68% ceiling. (2) **Focal & contrastive losses did NOT help** — both land *below* plain baseline (their DL members ~65% < baseline DL ~69%; the shared baseline sklearn carry them) — a clean negative ablation result. (3) Consistent with the rest of the audit: baseline 68.4% ≈ honest-text-only 68.6%, Model B 76.7% ≈ the corrected comparison. The frontend "Detector" dropdown serves all four for side-by-side inspection.

**Per-business win-rate (`per_business_winrate.py`).** Beyond the aggregate, Model B vs the text-only baseline *per business* (9-model ensemble, businesses with ≥40 reviews, n=36): **Model B wins 83%, ties 6%, loses 11%.** Wins are often large (+12 to +21 pp accuracy, e.g. zip_352 69→85%, zip_3215 66→83%, chi_res…GQ 67→88%). The minority where text-only wins (e.g. zip_3237 75→71%) are businesses where the review text happens to be especially discriminative — *not* leakage: checked `zip_3237` directly, only ~10% of its reviews have a near-dup in train and only 2% are internally templated, so its strong baseline number is genuine, not memorization. So per-business performance varies but **Model B leads the clear majority of businesses**, consistent with the +8 pp aggregate.

**Model B focal/contrastive DL variants (TUNED best-shot) — new ablation.** The text-only focal/contrastive above landed *below* baseline, which left open whether the losses were simply **under-tuned**. To give them a genuine best shot I built tuned **Model B** variants — same text+behavioral fusion, only the DL member's loss swapped — where each notebook **sweeps** the loss hyperparameters on a fast proxy (stratified subsample + short epochs, scored on the full balanced val) then retrains the winner on the full 642k. Standard Model B is untouched; the exported ONNX is a drop-in (the SupCon projection head is train-only). Notebooks: `train_yelp_multimodal_{lstm,distilbert}_{focal,contrastive}.ipynb`.

| Model B DL member | best config (swept) | test macro-F1 | fakeR / fakeP / genSpec | vs standard Model B member |
|---|---|---|---|---|
| **LSTM focal** | γ=1.0, α=0.75, thr 0.40 | **76.94%** | 79.6 / 75.5 / 74.3 | ≈ **tie** (standard LSTM 77.0%) |
| **DistilBERT focal** | γ=3.0, α=0.25, thr 0.30 | **77.22%** | 84.3 / 73.9 / 70.4 | ≈ **tie** (standard DistilBERT 77.3%) |
| **LSTM contrastive** | λ=0.5, τ=0.1, thr 0.50 | **77.20%** | 83.9 / 74.0 / 70.7 | ≈ **tie** (standard LSTM 77.0%) |
| **DistilBERT contrastive** | λ=0.1, τ=0.1, thr 0.50 | **76.39%** | 78.3 / 75.3 / 74.5 | slightly **below** (standard DistilBERT 77.3%) |

**Finding (so far):** both tuned Model B **focal** variants **tie their standard Model B counterparts** — LSTM 76.94% vs 77.0%, DistilBERT 77.22% vs 77.3% (both within run-to-run noise). This is a large jump over the *text-only* focal (DL ~65%, below baseline); the difference is the **retained behavioral modality** — in the multi-modal model the loss swap is ~neutral (focal adds no new signal, but no longer *hurts* once the behavioral branch carries the lift and α/γ are tuned). Notably the sweep chose **different optima per architecture and both far from the textbook γ=2 / α=0.89 default** — LSTM γ=1.0/α=0.75, DistilBERT γ=3.0/α=0.25 — directly vindicating per-model tuning over hard-set defaults (the earlier "below baseline" was partly an under-tuning artifact). The DistilBERT focal trades genuine specificity (70.4%) for high fake recall (84.3%), a different operating point than standard. The tuned **LSTM contrastive also ties** (77.20% vs standard LSTM 77.0%) — and its sweep chose **λ=0.5**, a strong contrastive weight (5× my original conservative λ=0.1), with the SupCon term genuinely active in the loss (~5.4), confirming the contrastive objective was doing real work rather than a token nudge. Honest read: tuning rescued both focal and contrastive from *below-baseline* (text-only) to *parity* (multi-modal), but neither **beats** standard Model B — the behavioral ceiling holds; the losses only reweight/reshape, they add no new signal.

**Final verdict (all 4 tuned variants done).** Three of four **tie** their standard Model B counterpart (LSTM focal 76.94 / DistilBERT focal 77.22 / LSTM contrastive 77.20, vs standard LSTM 77.0 / DistilBERT 77.3); the fourth, **DistilBERT contrastive, lands slightly below (76.39%)** — it overfits within one epoch (train CE 0.46→0.13 while val F1 falls 0.7585→0.7177; early-stop correctly restored epoch 1) and its sweep ranked **λ=0.1, the *weakest* contrastive setting, as best**, i.e. SupCon was net-unhelpful for the transformer. **Bottom line:** even with a genuine best-shot sweep (each variant's chosen config differs from the textbook γ=2/α=0.89/λ=0.1 default), focal and contrastive **match but do not exceed** standard Model B on the multi-modal task — a clean, fair negative ablation. The lift comes from the **behavioral modality**, not the loss function. (This corrected the earlier text-only-ablation impression that they were strictly *worse*; on equal, tuned footing they're *equal*, except DistilBERT-contrastive which the transformer's fast overfitting drags slightly under.)

*Ensemble note:* the 10-model Model B ensemble is **not separately reported per variant** because all three variants share the **identical 8 multi-modal sklearn** — only 2 of the 10 votes (the LSTM + DistilBERT members) change, and those are all ~76–77%, so every variant's served ensemble is pinned at standard's ~76.7% by construction. The DL-member macro-F1 above is the discriminating, decision-relevant number; a per-variant ensemble run was deemed redundant (it would re-measure ~76.7% three times). All three are still **served live** via the frontend Detector dropdown (Multi-Modal · Baseline / Focal Loss DL / Contrastive Learning DL) for side-by-side inspection.

### Honest caveats & limitations
- **Only the transformer generalizes OOD.** The sklearn/LSTM members are review-surface
  learners and are unreliable on novel formats (HC3 ~40–48%); the ensemble's OOD
  reliability rests on DistilBERT. This matters only for genuinely novel input — live
  uploads are **review-gated**, so they're in-distribution, where all members perform.
- **Persistent hard floor:** legacy products-only synthetic corpora (Salminen
  GPT-2-fine-tuned-on-reviews, sutro, Kenshiii) drive the low LOGO (0.15) / products-LOD
  (0.33) numbers. These are **in-distribution in deployment** (caught fine); the low
  scores only describe a never-seen *legacy* generator.
- **Model B is validated within Yelp, not cross-platform.** Its behavioral features are
  Yelp-specific aggregates (reviewer/business history); no external labeled behavioral-
  fake corpus exists to test cross-platform transfer, so its generalization is shown on
  held-out Yelp + unseen AI generators only — a stated boundary, not papered over.
- Small in-app spot-checks (e.g. a 24-row held-out real-vs-AI CSV → 23/24) are sanity
  checks, **not benchmarks** — cite the held-out splits, LOGO/LOD, and HC3, not demos.

---

## Model Performance

### Rule-Based Suspicion Scorer — evaluation on a hand-labeled set
The first stage of the Amazon/Steam fake pipeline is a rule-based heuristic (9 red-flag rules →
`suspicion_score` 0–100; see `score_reviews.py`). To quantify it on its own, I fetched Steam reviews
(Cyberpunk 2077 + other games), scored them, and **hand-labeled 170 reviews** (`prepare_labeling_batch.py`
→ `labeling_batch_{internal,blind}.csv` → `labeled_reviews.csv`; labels genuine/suspicious/uncertain =
70/14/86). Excluding "uncertain", **84 confidently-labeled reviews** (70 genuine / 14 suspicious) were used
to sweep the suspicion-score threshold (`analyze_rule_thresholds.py` → `rule_threshold_analysis.csv`;
`evaluate_rules.py` for the confusion sets `rule_{true,false}_{positives,negatives}.csv`):

| threshold | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| 10 | 60.7% | 0.30 | 1.00 | 0.46 |
| 20 | 82.1% | 0.48 | 0.71 | 0.57 |
| 25 | 88.1% | 0.70 | 0.50 | 0.58 |
| 40 | 88.1% | 0.75 | 0.43 | 0.55 |
| **45–50** | **90.5%** | **1.00** | 0.43 | **0.60** |
| 60 | 88.1% | 1.00 | 0.29 | 0.44 |

**Findings:** the rules are **high-precision, low-recall** — at the deployed-style threshold (45–50) they
flag only the clearest spam (precision 1.00, recall 0.43, F1 0.60, accuracy 90.5%: 6 TP / 0 FP / 70 TN /
8 FN). Dropping the threshold to 10 recovers every suspicious review (recall 1.00) but precision collapses
to 0.30. This confirms the design intent: the rules are a cheap, **explainable first-pass filter that
catches obvious fakes without over-flagging**, while the ML/DL ensemble carries the subtler cases.

**⚠️ Caveats:** small (n=84 confident labels), single domain (Steam games), single annotator, and **51% of
the batch was "uncertain"** (text alone is frequently inconclusive — itself a finding). Report this as a
sanity-check of the heuristic, **not** a benchmark.

### Sentiment Models — in-distribution (held-out 20% of the merged IMDB + tweets 500k sample), all 10 models
| Model       | Accuracy | Precision | Recall | F1    |
|-------------|----------|-----------|--------|-------|
| DistilBERT† | 86.0%    | 86.2%     | 85.6%  | 85.9% |
| LSTM†       | 81.1%    | 81.5%     | 80.6%  | 81.0% |
| LR          | 77.2%    | 75.9%     | 79.8%  | 77.8% |
| XGBoost     | 73.5%    | 69.9%     | 82.5%  | 75.7% |
| NB          | 75.5%    | 76.0%     | 74.4%  | 75.2% |
| RF*         | 72.2%    | 68.2%     | 83.1%  | 75.0% |
| MLP         | 73.2%    | 73.3%     | 72.9%  | 73.1% |
| DT          | 69.4%    | 69.1%     | 70.3%  | 69.7% |
| KNN         | 66.5%    | 65.8%     | 68.6%  | 67.2% |
| AdaBoost    | 57.4%    | 74.9%     | 22.4%  | 34.5% |

*RF retrained with bounded trees (`max_depth=40, min_samples_leaf=5`) to cut the model from 2.19 GB → 17 MB (127×). The original unbounded RF scored 75.2% / 74.6% / 76.6% / 75.6% (acc/prec/rec/F1); bounding cost ~3 pp accuracy with F1 unchanged — a deliberate size/RAM trade (see `shrink_sentiment_rf.py`).

†LSTM and DistilBERT were evaluated by `evaluate_sentiment_dl_indistribution.ipynb` (Colab GPU) on **their own** held-out 20% split — the same recipe/seed as the sklearn split (`SEED=42`, all 50k IMDB + tweets → 500k, 80/20 stratified, ~90:10 tweet:IMDB), independently sampled (the DL notebooks use pandas `.sample(random_state=42)`, the sklearn trainer uses Python `random.sample(42)` — different RNGs → different rows). So the rows differ, but **each model is scored on data it never trained on** (leakage-free; running the DL models on the *sklearn* rows would leak ≈¼ of them and inflate the scores). Same binary metrics as the sklearn rows (`pos_label=1`, `zero_division=0`), threshold 0.5 (LSTM `prob>0.5`, DistilBERT `argmax`) — as served. **DistilBERT leads in-distribution (85.9% F1) just as it leads cross-domain (next table); the LSTM is second (81.0%). "Logistic regression leads" holds only among the eight classical models** (n=100,000 held-out reviews; split fingerprint SHA1 `d43ac7d0…`).

### Sentiment Models — real-world benchmark on Amazon reviews (all 10 models, incl. the DL ones)
The table above is the in-distribution held-out test (merged IMDB+tweets, all 10 models). To benchmark the **served**
sentiment models on a **different domain** (real product reviews), `evaluate_sentiment_accuracy.py`
scores every model against **Amazon star ratings** on **10,000 Amazon Fine Food reviews** (1–2★ → negative,
4–5★ → positive). This is a **cross-domain** test (the models trained on IMDB + tweets, not product reviews).
→ `sentiment_evaluation_results.csv`:

| model | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| **DistilBERT** | **86.7%** | 90.4% | 82.1% | **86.0%** |
| Logistic Regression | 78.6% | 78.4% | 78.9% | 78.7% |
| **LSTM** | 77.0% | 77.8% | 75.6% | 76.7% |
| Naïve Bayes | 76.1% | 74.5% | 79.3% | 76.8% |
| MLP | 75.2% | 73.1% | 79.7% | 76.2% |
| Random Forest | 74.8% | 74.9% | 74.7% | 74.8% |
| XGBoost | 74.5% | 72.6% | 78.7% | 75.5% |
| Decision Tree | 63.2% | 62.6% | 65.5% | 64.0% |
| AdaBoost | 62.4% | 60.0% | 74.6% | 66.5% |
| KNN | 57.8% | 57.0% | 63.9% | 60.2% |

**Findings:** **DistilBERT clearly wins (86.7% / F1 0.86)** — ~8 pp above the best non-transformer (LR 78.6%)
and the only model that transfers strongly to the new domain. The LSTM (77.0%) sits with the linear/NB tier;
the tree-bag models and KNN trail (57–63%). This is the deployment-relevant sentiment number (real Amazon
reviews, not the training corpus), and it mirrors the fake-detector story: **the transformer generalizes best
across domains.** *(Caveat: a star rating is a proxy for sentiment — a 5★ review can still contain criticism —
so read ~87% as a strong real-world estimate, not a perfect-label benchmark.)*

### Yelp Fake Detection — Deduped Re-run (collection-duplicate removal + maximised models)

**This is the current, authoritative Yelp results section.** All subsections below it (the
252,540-row "Final Training Run", focal/contrastive ablation, preliminary runs) are *earlier*
runs on the pre-dedup split, retained for reference — not overwritten.

**What changed vs the earlier runs:**
1. **Collection-duplicate removal.** The raw sources contain the *same physical review* copied
   across overlapping files (YelpNYC ⊂ YelpZip). `prepare_yelp_split.ipynb` and
   `prepare_yelp_split_imbalanced.ipynb` now drop duplicates on **(normalized text, rating, date)**,
   removing 354,611 copies (34.3%) from the 1,035,045 raw pool → 680,434 unique reviews (89,790 fake, 13.2%).
   Natural duplicates (different reviewers writing the same short text, e.g. "good") differ in
   rating/date and are preserved (675 remain in the balanced set).
2. **Maximised models (same recipe across baseline/focal/contrastive for fair ablation):**
   - DistilBERT: MAX_LEN 128→256 (~44% of reviews were truncated at 128), layer-wise LR decay, cosine schedule, early stopping on val macro F1.
   - LSTM: MAX_LEN 150→250 (~21% truncated at 150), GloVe 100d→300d, BiLSTM 64→128.

**Balanced split (`yelp_split.csv`):** 179,580 rows (train 143,670 / val 17,963 / test 17,947), exact 50/50, grouped by normalized text → 0% leakage verified.

**Reading the numbers:** Test macro F1 is on the balanced 50/50 set. "Fake P@13%" is the
projected fake-class precision at the real-world ~13% fake prior (recall/FPR are prior-independent;
precision is re-projected) — the honest deployment view. `evaluate_yelp_ablation.py` prints this.

**Final balanced-test ablation** (`evaluate_yelp_ablation.py` — threshold re-tuned on the balanced val for every model; PR-AUC/ROC-AUC are threshold-free). Test n=17,947 (50/50). Sorted by macro F1:

| Model                    | Thr* | Macro F1 | Fake F1 | PR-AUC | ROC-AUC | Fake P@13% |
|--------------------------|------|----------|---------|--------|---------|------------|
| DistilBERT (focal γ=2)   | 0.49 | 69.6%    | 69.1%   | 75.0%  | 76.8%   | 26.1%      |
| DistilBERT (baseline)    | 0.48 | 69.6%    | 70.5%   | 74.5%  | 76.3%   | 24.6%      |
| DistilBERT (contrastive) | 0.45 | 68.9%    | 69.9%   | 73.8%  | 75.7%   | 24.0%      |
| LSTM (baseline)          | 0.48 | 68.5%    | 69.5%   | 72.9%  | 75.1%   | 23.7%      |
| LSTM (focal γ=2)         | 0.47 | 68.3%    | 69.2%   | 72.2%  | 74.6%   | 23.5%      |
| LSTM (contrastive)       | 0.51 | 68.2%    | 69.2%   | 73.2%  | 75.1%   | 23.5%      |

**Delta vs same-architecture baseline** (variant − baseline): LSTM focal −0.2% F1 / −0.6% PR-AUC; LSTM contrastive −0.3% / +0.3%; DistilBERT focal +0.0% / +0.5%; DistilBERT contrastive −0.7% / −0.7%.

**Findings (this is the authoritative comparison — supersedes the "projected/pending" remarks in the per-model notes below):**
- **DistilBERT > LSTM by ~1pp** on every metric. Consistent with the pre-dedup runs.
- **Neither focal nor contrastive beats its baseline** — all deltas are within ±0.7pp (run-to-run noise). On equal footing (balanced test + balanced-tuned thresholds) the imbalance techniques **tie** the baseline; they do not lift the text-signal ceiling.
- **Threshold re-tuning was essential for fairness:** on the imbalanced internal test the focal/contrastive models read ~65–66% (thresholds tuned for ~21% fake); re-tuned on the balanced val they read 68–69%. The earlier apparent gap was largely a threshold-regime artifact, not model quality.
- **Best single detector: DistilBERT focal** — ties the baseline on macro F1 (69.6%), edges it on threshold-free PR-AUC (75.0%) and deployment precision (26.1% @13%, lowest FPR 28.7% → fewest false flags ~250 vs ~290–300 per 1,000). The baseline trades that for higher fake recall (72.5% vs 68.0%).
- **All six sit in a ~1.4pp macro-F1 band (68.2–69.6%)** — strong confirmation the ceiling is the behavioural labels, not the architecture or loss.
- **Deployment reality:** even the best model flags ~340/1,000 reviews at the real 13% prior with only ~26% precision → not a standalone detector; corroborate with behavioural signals.

### Multi-Modal Yelp Fake Detection — text + behavioral fusion (`train_yelp_multimodal_fusion.py`)

The six text-only models above all sit in a 68.2–69.6% macro-F1 band — the ceiling is the *behavioural* nature of Yelp's labels (its spam filter uses reviewer/metadata signals), not architecture or loss. This operationalises the "corroborate with behavioural signals" note above as a **single multi-modal model** that ingests both the review **text** and engineered **behavioural** features and emits one fake/genuine verdict (the thesis's "multi-modal" framing).

**Behavioural features (17)** — `build_yelp_multimodal_features.py` recovers metadata by joining `yelp_split.csv` back to the raw YelpZip/NYC/Chi files (**100% match**, 178,905 rows) and aggregates over the **full 1.03M-review pool** (not the balanced subsample, so counts/rates are real): reviewer review-count, reviews/day (velocity), max-reviews-in-a-day (burst), rating mean/std, %positive, %extreme, singleton flag; this review's rating, extreme flag, deviation from the business mean; business review-count, rating mean/std; review length (words/chars). `user_reviews_on_this_biz` dropped (constant 1 — Yelp allows one review per user per business). Class separation is strong: singleton-reviewer rate 0.67 (fake) vs 0.31 (genuine), reviews/day 2.2× higher for fakes, reviewer review-count ~2.3 vs ~11.

**Setup:** **identical** train/val/test split as the text-only baselines (keyed on normalised text → 143,123 / 17,891 / 17,891; this collapses the 675 natural-duplicate texts vs the deep models' 17,947-row test, otherwise the same set, zero leakage). Text modality = TF-IDF (1–2gram, stopwords stripped) + LogisticRegression, with train-row text scores produced **out-of-fold** (`cross_val_predict`, cv=5) so the fusion never sees a text score fit on its own row. Behavioural-only and fusion classifiers = XGBoost (400 trees, depth 6). Thresholds tuned on val for macro-F1, reported on test.

| Model | Macro F1 | Fake P | Fake R | Acc |
|---|---|---|---|---|
| text-only (TF-IDF+LR) | 67.5% | 66.7% | 69.8% | 67.6% |
| behavioural-only (XGB) | 75.9% | 72.2% | 84.7% | 76.1% |
| **FUSION (multi-modal, XGB)** | **77.2%** | 73.8% | 84.6% | 77.3% |

**Findings:**
- **+9.7pp macro-F1** from text-only → fusion on the identical test set; the fusion also clears the **best** text-only deep model (DistilBERT focal, 69.6%) by **~7.6pp**.
- The lift is overwhelmingly behavioural — **behavioural-only alone hits 75.9%**, confirming the text-only plateau is a *modality* ceiling, not a task ceiling.
- But the model is genuinely multi-modal: **`text_score` is the 2nd-most-important feature** (behind `user_reviews_per_day`), and fusion beats behavioural-only by +1.3pp — so the text modality still contributes after behaviour is accounted for. Top features: reviews/day (burstiness) → text_score → reviewer review-count → %extreme-ratings.
- This is **Model #1 of 2** (feature-level / late fusion, trained locally). Model #2 = a **deep joint multi-modal net** (learned text-encoder branch + behavioural branch, trained end-to-end on Colab) — pending; expected to raise the text contribution above the TF-IDF+LR branch used here.

### Generalized (catch-all) Multi-Modal Detector — AI-augmentation (`build_yelp_multimodal_augmented.py` + `train_yelp_multimodal_augmented.py`)

The Model #1 detector catches *human behavioural* fakes only. Phase 2 makes the **same single model** also catch *AI-generated* reviews, by injecting AI-generated restaurant reviews (restaurant domain chosen to match Yelp's domain) as additional fakes (label 1). Final AI pool: **9,532 reviews across 5 generator families** — Qwen / Llama-3 / Zephyr / Yi (open, `generate_ai_restaurant_reviews.ipynb`) + GPT-4o-mini (existing).

**Design (why it's honest):** each AI fake is given a behavioural feature vector **copied from a randomly-sampled genuine Yelp reviewer** — only the two text-length features are overwritten with the AI text's real length. So an AI fake is *behaviourally indistinguishable from a genuine reviewer*: the behavioural branch is deliberately blinded to it, and the only way to flag it is from the **text**. This forces the text branch to learn the real AI-writing signal. Same-domain (restaurant) AI vs genuine → the text branch learns *style*, not topic. Rows tagged `fake_type` ∈ {genuine, human_fake, ai_fake}.

**Length-shortcut found & fixed (important):** the first AI generation produced only *long* reviews (~190 words); real Yelp reviews are short-to-medium (median ~75). That made **review length alone separate AI from real at AUC 0.81** — a shortcut (the model would catch AI for being long, not for *being AI*). Fixed two ways (defense-in-depth): (1) generation now draws from **length buckets weighted to the real Yelp distribution** (median 81, 24% >150w ≈ Yelp's 22%); (2) the builder **length-stratifies the AI pool** to match Yelp per word-length bin. Result: **length AUC → 0.49** (no signal). AI-recall *rose* (90.1%→92.5%) after the fix, confirming it was genuine style detection, not length.

**Test results** (augmented test: 8,969 genuine / 8,922 human-fake / **1,009 AI-fake**). Fusion **upweights AI fakes ×10** (`AI_WEIGHT`) so the behavioural branch (which sees AI as behaviourally genuine) can't drown out the text signal:

| Model | Macro F1 | Human-fake recall | AI-fake recall | Genuine spec |
|---|---|---|---|---|
| text-only (TF-IDF+LR) | 69.0% | 67.7% | **97.2%** | 67.3% |
| behavioural-only | 73.3% | **87.0%** | 39.5% | 64.4% |
| **FUSION (generalized, AI wt 10)** | **75.9%** | 79.5% | **92.5%** | 70.9% |

**Findings — the multi-modal generalization result:**
- **Each modality does one job:** behavioural-only catches human spam (87.0%) but is near-blind to AI (39.5% — genuine-sampled behaviour worked); text-only catches AI (97.2%) but is weak on human spam (67.7%).
- **Only the fusion is general:** human fakes 79.5% **and** AI fakes **92.5%** (behavioural-only's 39.5% → 92.5%), macro-F1 75.9%, genuine spec 70.9%. AI detection cost ~nothing on the human-fake task (vs Model #1's 77.2%).
- **Length is no longer a crutch** (AUC 0.49) — the 92.5% AI-recall is genuine writing-style detection.
- **Per-generator AI recall (in-distribution), tight across all 5 families:** llama3 94.6%, gpt 92.9%, zephyr 92.6%, yi 92.4%, qwen 89.9%.
- **Leave-one-generator-out** (`evaluate_yelp_ai_logo.py`, one generator fully held out): **mean unseen-generator recall 86.9%** (zephyr 91.3%, gpt 89.2%, yi 87.2%, llama3 83.9%, qwen 82.9%). The in-dist→LOGO gap is only ~6pp (92.5→86.9) → the model learned general "machine-ness", not per-generator fingerprints (vs the earlier 4-generator/length-inflated run which had a 16pp gap and 74.5% mean). Models saved to `data/yelp_multimodal_generalized_models/`.

### Full Model-B ensemble — 8 sklearn + LSTM + DistilBERT, each multi-modal

To mirror the Amazon/Yelp sections (per-model dot-strip), Model B is delivered as the **full 10-model ensemble**, every member multi-modal (text + behaviour), trained on the AI-augmented CSV with AI ×10 up-weighting and val-tuned thresholds:

- **8 sklearn** (`train_yelp_multimodal_sklearn.py` → `data/yelp_multimodal_sklearn_models/`): each is a `ColumnTransformer` Pipeline — **TF-IDF(text) ⊕ MinMax(17 behavioural features)** → classifier (MinMax keeps it non-negative for MultinomialNB). Test results:

  | model | macro-F1 | human-fake R | AI-fake R | genuine spec |
  |---|---|---|---|---|
  | xgb | 75.2% | 78.2% | 98.2% | 70.1% |
  | rf* | 74.0% | 77.5% | 95.8% | 68.6% |
  | mlp | 74.1% | 76.2% | 95.6% | 70.0% |
  | lr | 74.0% | 73.5% | 99.1% | 71.9% |
  | nb | 68.0% | 68.7% | 99.9% | 64.1% |
  | knn | 68.0% | 74.2% | 72.2% | 61.9% |
  | dt | 67.4% | 69.2% | 78.9% | 64.6% |
  | adaboost | 67.1% | 62.9% | 78.1% | 69.9% |

  *rf retrained depth-bounded (`max_depth=40, min_samples_leaf=5`, `shrink_rfs.py`) → **531 MB → 55 MB (10×)**; macro-F1 75.1→74.0 (−1.1pp), trading ~4pp human-fake recall for +4pp AI-fake recall — negligible for the consensus.

  Each model genuinely fuses both modalities — high AI recall (TF-IDF sees AI style directly: lr/nb/xgb/mlp 95–99%) and solid human-fake recall (rf 78%, xgb 78%). xgb (75.2%) and rf (74.0%, depth-bounded) match the XGBoost feature-fusion model (75.9%); dt/adaboost/knn sit lower (same spread as the other sklearn sections).
- **LSTM fusion** (`train_yelp_multimodal_lstm.ipynb`, Colab): GloVe-BiLSTM text branch + behavioural MLP, fused end-to-end. **Test: macro-F1 76.5%, human-fake 77.2%, AI-fake 99.3%, genuine-spec 73.4%** (best epoch 11/30, early-stopped on val macro-F1 0.766). Edges the sklearn members (xgb/rf 75%) and the XGBoost feature-fusion (75.9%); near-perfect AI recall.
- **DistilBERT fusion** (`train_yelp_multimodal_distilbert.ipynb`, Colab): DistilBERT [CLS] (768-d) → projection, fused with behavioural MLP, end-to-end (66.6M params); recipe = LLRD + 10% warmup/linear-decay + grad-clip + AMP. **Test: macro-F1 76.8%, human-fake 78.9%, AI-fake 99.6%, genuine-spec 72.6%** (best epoch 2/5, early-stopped — DistilBERT peaks fast then overfits). **Best member of the ensemble.**

**Ensemble summary (all 10 members, macro-F1 on the same test set):** DistilBERT **76.8%** > LSTM 76.5% > xgb 75.2% > mlp 74.1% ≈ rf 74.0% ≈ lr 74.0% > nb 68.0% ≈ knn 68.0% > dt 67.4% > adaboost 67.1%. The two deep fusion models edge the sklearn members by ~1.5pp; **AI-fake recall is ~99% across lr/nb/LSTM/DistilBERT** (the text branch detects AI strongly), while behavioural features carry the human-spam side. All cluster in a ~67–77% band — the multi-modal lift over the 66–69% text-only ceiling holds across every architecture.

All three trainers consume `data/yelp_multimodal_augmented.csv` (audited: 188,437 rows, 0 NaN/inf, 0 duplicate texts, **0 train/test leakage**, length-AUC 0.49, 5 generator families). DL models export to ONNX (`data/yelp_multimodal_lstm_onnx/`, `data/yelp_multimodal_distilbert_onnx/`) for thread-safe serving like the project's other DL models.

### Graph Neural Network — relational signal for coordinated rings (advisor suggestion)

A GNN over the **reviewer ↔ review ↔ business** graph, to catch **coordinated rings** that per-review models can't see. Graph built by `build_yelp_graph.py`: tripartite heterogeneous, **full graph (1,035,045 review + 487,897 reviewer + 6,168 business nodes) with partial labels** on the 178,905 `yelp_split` reviews — the **same train/val/test split as Model B + the text-only baseline** (test n = 17,891), so it is directly comparable. Trained (`train_yelp_gnn.ipynb`, Colab GPU, ~minutes) as a **2-layer heterogeneous GraphSAGE** (hidden 128), class-weighted, early-stopped on val macro-F1 (best epoch 111/126). Node features are **behavioral + structural only — NO text** (the GNN's contribution is the relational signal); review (6) + reviewer (8) + business (3) features.

| detector (same clean Yelp test, n=17,891) | modality | macro-F1 |
|---|---|---|
| text-only baseline | text | ~68–69% (ceiling) |
| **GNN** | **graph structure + behavioral, NO text** | **74.7%** (fake recall 82.6% / precision 71.5% / acc 74.9%) |
| Model B | text + behavioral | 74–77% |

**Finding:** using **zero text** — purely who-reviews-what structure + behavioral features — the GNN reaches **74.7% macro-F1**, ~6 pp above the text-only ceiling and squarely in Model B's text+behavioral band. This is the thesis's **third, orthogonal modality** (text → Model A; text+behavioral → Model B; **pure relational → GNN**): it confirms Yelp fraud is fundamentally **relational/behavioral, not textual** (structure alone matches text+behavioral), and it is the only detector of the three that can flag **coordinated rings** as a structural pattern. *(v1 GNN uses no text features — adding a DistilBERT [CLS] embedding per review node is a clean future extension.)*

**Coordinated-ring extraction (`extract_yelp_rings.py`).** Beyond per-review classification, we extract the rings themselves structurally from the full raw data (1.03M reviews): two FAKE reviewers are linked iff they jointly fake-reviewed **≥3 of the same businesses** (mega-target businesses with >30 fake reviewers are skipped as noise); connected components of ≥4 reviewers = rings. Result (correct labels — **note:** YelpZip/NYC use `-1`=fake / `1`=genuine; an earlier figure of "16 rings" came from an inverted-label bug and was actually genuine reviewers): at strict min-shared 3 the full raw data yields **2 rings (10 + 5 reviewers, 100% fake)**; counts rise as the linking is loosened (the ring *count* is parameter-dependent, not a canonical truth). Saved to `data/yelp_rings.csv`. This relational/coordination structure is exactly what the per-review text (Model A) and text+behavioral (Model B) detectors structurally **cannot** surface — it is the GNN's distinctive contribution.

**GNN-driven, label-free ring detection (`extract_rings_gnn.py`) — the deployable version.** The rings above use ground-truth labels; in production there are no labels, so the GNN must supply them. Extracting rings from the GNN's *predictions* (no labels): at its default 0.5 threshold the GNN over-flags **38%** of reviews (a calibration artifact of balanced 50/50 training) — too diffuse, rings collapse to 0. But thresholded to flag **~13%** (≈ the true fake rate, confidence ≥ 0.7) it surfaces **2 coordinated rings whose members are 96% genuinely fake — using NO labels**, recovering essentially the label-based rings on that graph. This is the GNN's genuine, deployable contribution: **label-free** coordinated-ring detection. **Honest precision (corrected):** an earlier "88% precision" figure validated only against the *labeled subset* (~half the ring members); validated against **all** labels (via `graph_full`), the balanced GNN surfaces **~27 rings / ~296 reviewers at ~55% precision** when thresholded to the true 12% fake rate. On a 12% base rate that is **~4.5× enrichment** — a real but **modest** signal, not high-precision; the GNN's predictions on *unseen* reviewers (the previously-unlabeled half) are markedly weaker than on training-distribution reviewers.

**Focal-loss / full-data experiment (negative result, `train_yelp_gnn_focal.ipynb` + `build_yelp_graph_full.py`).** To try to fix the balanced-training over-prediction, the GNN was retrained on **all 1.03M reviews (true 12% fake)** with **focal loss** (γ=2, inverse-freq α). It did **not** help: classification macro-F1 **73.97% ≈ baseline 74.7%** (data ceiling holds); calibration got *worse* (over-flags 40%, probabilities compressed into 0.45–0.73 with no high-confidence band); and on the **full-label** ring test (all configs thresholded to the true 12%) **no focal config beats the balanced GNN**: best focal merely ties it (focal CE-invfreq ~56% precision / 24 rings vs balanced ~55% / 27 rings) or trades precision for count (focal α=[1,4]: 37 rings but only 38% precision); the rest collapse to 0 rings. There is a hard **calibration-vs-accuracy tension** (the best-calibrated config, focal-no-α at 19.7% predicted-fake, has only 59.9% macro-F1). **Conclusion: properly-swept focal loss / class-rebalancing does not improve the GNN** — for classification or rings — on this behaviorally-labeled data.

**Reviewer↔reviewer co-edges (positive result, `build_reviewer_edges.py` + `train_yelp_gnn_reviewer_edges.ipynb`).** The base graph is `reviewer→review→business`, so two co-reviewing reviewers sit **4 hops apart** — a 2-layer GNN cannot reach across to see their coordination. We added an explicit **`reviewer↔reviewer` relation**: link two reviewers iff they jointly reviewed **≥2 of the same businesses** (`reviewer_edges.npz`: 96,669 edges over 16,047 reviewers, mean degree 12), making coordination a **1-hop** neighbour signal. Trained two models identically (same seed, CE inverse-frequency, full 1.03M graph) differing **only** in the presence of co-edges, and compared rings on the **full labels** (both thresholded to the true 12.2% fake rate, so only graph *structure* differs — `compare_reviewer_edge_rings.py`):

| variant | rings | reviewers | review-precision | reviewer-precision |
|---|---|---|---|---|
| BASE (no co-edges) | 32 | 285 | 51% | 51% |
| **CO (reviewer↔reviewer)** | **34** | **358** | **62%** | **55%** |

**Classification is unchanged** (macro-F1 74.6% → 74.7% — the behavioral-label ceiling is data-bound and no graph edge moves it). On this (buggy-graph) run the co-edges *looked* like a large ring win — +26% more reviewers and +11pp precision (62% vs 51%). **⚠️ This was partly an artifact** — see the correction below: once a graph bug was fixed the co-edge benefit shrank to ~+4pp. The structural intuition (1-hop coordination beats the unreachable 4-hop path) still holds, just more modestly than this table suggested.

**Free ring-extraction tuning (`sweep_ring_extraction.py`).** The deployable union-find extractor (run on the GNN's flagged reviews) has a tunable **link-strictness** knob, `MIN_SHARED` = how many shared flagged businesses link two reviewers. Sweeping it on the co-edge GNN gives a clean **precision/coverage curve** (no retrain): `MIN_SHARED=2` → broad (34 rings / 358 reviewers / ~58–62% precision); `MIN_SHARED=3` → high-precision (≈76–78% review-precision, ~80–86% reviewer-precision, but only ~30 reviewers). So the extractor is a tunable triage tool, not a single operating point — analysts can dial coverage vs precision.

**Chi reviewer bug fix + temporal-burst co-edges (corrected-graph re-run, `build_reviewer_edges_temporal.py` + `train_yelp_gnn_temporal.ipynb`).** Two changes:
1. **Bug fix** — `build_yelp_graph_full.py` was using the YelpChi *review-id* (`p[1]`) as the reviewer instead of the reviewer-id (`p[2]`), so every chi review became a unique singleton reviewer. Fixing it collapsed **28,405 bogus reviewers** (487,897 → 459,492) and let chi contribute real coordination structure. **This was the bigger win: classification rose 74.7% → 75.6%** (+1pp, the first thing to move it) and the corrected BASE graph alone yields an excellent high-precision ring mode (78% review / 86% reviewer at `MIN_SHARED=3`).
2. **Temporal-burst co-edges** — link reviewers only if they share ≥2 businesses **and** co-reviewed ≥1 within 30 days (12,277 edges vs 96,904 plain). **Negative result:** over-sparsified — *fewer* ring reviewers (202 vs 322) with no precision gain. Dropped.

Corrected-graph ring comparison (all CE inverse-frequency, same seed; full labels, thresholded to true 12.2%):

| `MIN_SHARED` | variant | rings | reviewers | review-prec | reviewer-prec |
|---|---|---|---|---|---|
| 2 (broad) | BASE | 21 | 301 | 54% | 52% |
| 2 (broad) | **CO (plain)** | 29 | 322 | **58%** | 54% |
| 2 (broad) | CO-temporal | 20 | 202 | 55% | 55% |
| 3 (high-prec) | **BASE** | 3 | 36 | **78%** | **86%** |
| 3 (high-prec) | CO (plain) | 3 | 39 | 68% | 82% |
| 3 (high-prec) | CO-temporal | 2 | 23 | 70% | 74% |

**Honest conclusions:** (a) the **chi fix** is the real improvement — +1pp classification and a strong corrected graph; (b) **plain co-edges** give a small, reliable broad-coverage gain (+4pp at `MIN_SHARED=2`, 58% vs 54%) — the earlier "+11pp" was inflated by the buggy weak BASE; (c) **temporal-burst edges do not help** (too sparse); (d) the `MIN_SHARED=3` numbers rest on 2–3 rings / 23–39 reviewers — too small to rank from a single seed (would need multi-seed to call BASE-vs-CO there). **Recommended config: corrected graph + plain co-edges**, served as a tunable triage tool (broad ~58% / high-precision ~70–78%, ~5–6× the 12% base rate). The GNN remains a *modest, honest* relational signal — useful for triage, not a high-precision standalone detector. **⚠️ Superseded by the cross-dataset dedup below — these ring numbers were partly inflated by YelpZip/NYC duplication.**

### Cross-dataset dedup — YelpNYC is ~99% a subset of YelpZip (the real data-integrity fix)

Cross-dataset overlap analysis of the raw review texts: **YelpNYC is 98.7% a subset of YelpZip** — 353,450 of YelpNYC's 358,016 unique texts are *already* in YelpZip (the two were collected separately, under independent id schemes). YelpChi is essentially disjoint (67,302 of 67,359 unique; NYC∩Chi outside Zip = 0). So the "full 1.03M" Yelp corpus is really **~678,663 unique reviews**. The prior GNN graph (`build_yelp_graph_full.py`, no cross-dataset dedup) double-counted ~353k reviews and split each cross-dataset reviewer into separate `zip_*`/`nyc_*` nodes.

**Fix:** added cross-dataset normalized-text dedup (keep-first → ALL Zip + ALL Chi + NYC-unique only), *before* the aggregates so reviewer/business counts aren't inflated. Graph drops **1,035,045 → 678,663 reviews** and **459,492 → 302,821 reviewers (−156,671 phantom duplicate reviewers)**. Retrained (`train_yelp_gnn_temporal.ipynb`, same notebook, deduped files):

| metric | undeduped (corrected) | **DEDUPED** |
|---|---|---|
| classification macro-F1 | 75.6% | **76.3%** (+0.7pp) |
| CO-plain rings (ms=2) review-prec | 58% / 322 reviewers | **49% / 149** |
| CO-temporal rings (ms=2) | 55% / 202 | **60% / 162** |
| BASE rings (ms=3) | 78% / 36 | **0 rings** |

**Key finding — the prior ring numbers were substantially INFLATED by the duplication.** A real reviewer's `zip_*` and `nyc_*` copies reviewed the **same businesses** (same reviews), so they formed self-duplicate "rings" of two — *one person* masquerading as a coordinated pair, and (if flagged fake) a 100%-fake "high-precision ring." Deduping collapses these artifacts: ring reviewers drop (322→~160), broad precision drops (58→49%), and **the 78% high-precision (ms=3) mode collapses to ~0 rings** — i.e. that impressive 78% was largely the self-duplicate artifact, not genuine coordination. **Honest ring-detection performance is ~48–60%, not 58–78%.** Classification *improved* slightly (+0.7pp) from removing the phantom reviewers. One flip: on the clean graph **CO-temporal is now marginally best** (60% vs CO-plain 49% / BASE 48% at ms=2) — opposite the undeduped negative result — but on small samples (16–23 rings / ~150 reviewers), so suggestive only. **Net: the GNN's relational signal is real but more modest than the undeduped numbers implied; the duplication was flattering it.** *(Note: the 178k `yelp_split` text models were unaffected — yelp_split was already deduped by text; only the GNN graph + behavioral aggregates, built on the raw union, had the duplication.)*

### GNN on a SECOND dataset — Amazon (Musical Instruments) benchmark

To show the GNN ring approach generalizes beyond Yelp, we added the standard **Amazon (Musical Instruments)** fraud-graph benchmark (CARE-GNN / PC-GNN / GADBench). Unlike Amazon Fine Foods (no labels, no AI text, ~52 coordinated users — verified unsuitable, see the "Why a GNN doesn't fit Amazon Fine Foods" analysis), this benchmark **ships fraud labels**: 11,944 reviewer nodes, 25 handcrafted features, 3 relations (`net_upu` shared-product, `net_usu` same-star-same-week, `net_uvu` text-similarity), label = paid fake reviewer. **Labels are a helpful-vote-ratio proxy**, not ground-truth fraud flags: reviewers with >80% helpful votes → benign, <20% → fraud, and the **20–80% middle band (plus those with too few votes) is left UNLABELED** — those are the first 3,305 nodes (placeholder `0` in the `.mat`, excluded from train/val/test by the CARE-GNN convention). They stay in the graph as unlabeled nodes so message passing can still flow *through* them between labeled reviewers (semi-supervised); they're never scored. The labeled set (3305+) is **9.5% fraud** (all 821 fraud labels lie there). Downloaded as `Amazon.mat` (CARE-GNN repo).

**Model (`train_amazon_gnn.py`, local CPU — the graph is tiny):** 2-layer multi-relation **mean-aggregation GraphSAGE** (pure torch, same relational-GraphSAGE family as the Yelp GNN), class-weighted CE, early-stopped on val macro-F1, **80/10/10 split of the labeled nodes** (semi-supervised: all nodes pass messages). **Test (n=864): macro-F1 90.9% · fraud recall 82.9% · fraud precision 84.0% · AUC 0.940.**

**Coordinated rings (`render_amazon_ring_levels.py`):** flag the top ~9.5% by P(fraud), induce the flagged-reviewer subgraph of `net_upu`, take its **k-core** (k = how many flagged co-reviewers each member connects to — the Amazon analog of Yelp's "shared businesses" strictness, since `net_upu` is binary and collapses product counts). Validation on labeled members only:

| level (k-core) | rings | reviewers | fraud-precision |
|---|---|---|---|
| Broad (k=1) | 28 | 262 | **86%** |
| Balanced (k=2) | 18 | 210 | **88%** |
| Strict (k=3) | 18 | 204 | **88%** |

**~9× enrichment over the 9.5% base rate** — far higher-precision rings than Yelp's ~58%.

**⚠️ Honest caveat (why Amazon scores so much higher than Yelp):** the Amazon benchmark's labels are a **proxy** (derived from helpful-vote ratios) and its 25 handcrafted features partly encode behaviour that correlates with that proxy — so the task is **more separable** than Yelp's behavioural spam-filter labels. The high numbers (0.94 AUC, 86–88% ring precision) are **in line with the published CARE-GNN/PC-GNN literature** on this exact benchmark, not a sign of a bug — but they reflect an easier, proxy-labelled task, so Yelp (harder, behaviour-labelled) and Amazon (easier, proxy-labelled) should be read as two *different* difficulty regimes, not a like-for-like comparison. Both are served in the frontend's **GNN Rings** section (dataset selector + precision/coverage selector).

**Why NOT a single combined Amazon+Yelp GNN (methodology note, for the defense).** Tempting ("more data, one model"), but unsound here: (1) the two graphs are **disjoint** — no edges connect Amazon reviewers to Yelp reviewers, so message passing never mixes them; a "combined graph" is just two disconnected components and only shares model *weights*, not structure. (2) **Incompatible schemas** — Yelp is tripartite reviewer↔review↔business with 6/8/3-dim behavioral features; Amazon is single-node-type with 3 user–user relations and 25 handcrafted features — no common input space without meaningless padding. (3) **Different label semantics** — Yelp = behavioral spam-filter (~13%) vs Amazon = helpful-vote proxy (~9.5%); one classifier on both = inconsistent supervision and risk of negative transfer (the easier Amazon proxy distorting the harder Yelp task). (4) The fraud-GNN literature (CARE-GNN, PC-GNN, GADBench) **always reports per-dataset**, never a concatenation. **Decision: keep them separate** and frame the result as *cross-platform generalization of the method* (same GNN approach, two independent benchmarks, per-dataset numbers) — a stronger, honest claim than a forced combined graph. A more interesting future angle would be cross-dataset *transfer* (train on one, test on the other), but the feature/label mismatch blocks it unless both graphs are rebuilt from raw with a shared schema.

**LSTM baseline — details:**
- BiLSTM(128) + GloVe 300d (trainable), MAX_LEN 250, 15.46M params (15M = 50k×300 embedding).
- Trained healthily: val loss bottomed at epoch 3 (0.608), overfit after; early stopping restored epoch 3; ReduceLROnPlateau fired at epoch 5.
- Threshold 0.51 (≈0.5) → well-calibrated, no class bias. Up ~1pp from the old LSTM (68% @ 0.57) and now matches the old DistilBERT baseline (69%).
- **Deployment @13% prior:** fake precision ≈ 24% — per 1,000 live reviews it flags ~368, of which only ~90 are truly fake and ~278 are genuine reviews wrongly flagged. Healthy classifier at the text ceiling, **not** a standalone deployment detector → corroborate with behavioural signals.

**LSTM focal (γ=2, α=0.878) — details:**
- Same architecture as baseline (BiLSTM 128 + GloVe 300d, MAX_LEN 250); trained on the imbalanced split (`yelp_split_imbalanced.csv`).
- Focal loss values small (~0.03–0.04, expected). Peaked epoch 2, early stopping restored epoch 2; ReduceLR fired epoch 4.
- Val macro F1 **64.9%** @ threshold 0.53 — on the **imbalanced** val, so NOT comparable to the baseline's balanced-val 68.8%.
- **Internal imbalanced test (~21% fake — NOT the balanced table above):** Genuine P/R/F1 0.86/0.81/0.83, Fake P/R/F1 0.42/0.51/0.46, accuracy 74%, macro F1 65%.
- More conservative than baseline: genuine recall 0.81 (vs 0.68), fake recall 0.51 (vs 0.69) → fewer false positives, fewer catches. Rough deployment @13% (TPR 0.51, FPR 0.19): fake precision ≈ 29% (indicative only — measured on a different test composition than baseline).
- Essentially unchanged vs the old focal LSTM (val 64.7%, fake F1 0.45) — the 300d/250-token levers barely move it; behavioural-label ceiling again.
- **Balanced-test macro F1 + Fake P@13% are pending `evaluate_yelp_ablation.py`** — the only apples-to-apples comparison against the baseline. The table row above stays "—" until then.

**LSTM contrastive (two-stage SupCon) — details:**
- Encoder BiLSTM 128 + GloVe 300d, MAX_LEN 250 (15.44M params); trained on the imbalanced split.
- **Stage 1 (SupCon, embedding frozen, BiLSTM+projection trained):** loss 5.529 → 5.486 over 10 epochs. Random baseline for batch=256 is log(255)≈5.541, so it moved ~0.055 below random — about 2× the old run's drop (5.533→5.515), thanks to the richer frozen GloVe 300d, but still tiny → weak text signal for behavioural labels confirmed again.
- **Stage 2 (classifier on SupCon-pretrained encoder, balanced batches):** peaked epoch 2, early stopping restored epoch 2. Val macro F1 **65.4%** @ threshold 0.65 (on the **imbalanced** val — NOT comparable to baseline's balanced 68.8%).
- **Internal imbalanced test (~21% fake — NOT the balanced table above):** Genuine P/R/F1 0.86/0.82/0.84, Fake P/R/F1 0.43/0.50/0.46, accuracy 75%, macro F1 65%.
- Marginally better than the old contrastive LSTM (val 64.7%→65.4%, fake F1 0.45→0.46) and ~ties the new focal LSTM (val 64.9%). Projected to a balanced 50/50 test (fake recall 0.50, FPR 0.18) ≈ 65% macro F1 → still **below** the baseline's 69%.
- **Balanced-test macro F1 + Fake P@13% pending `evaluate_yelp_ablation.py`.** Table row stays "—" until then.
- ⚠️ **Threshold-regime caveat:** its 0.65 threshold was tuned on the 21%-fake val. Applied as-is to the balanced test it is miscalibrated and will understate balanced macro F1. `evaluate_yelp_ablation.py` now re-tunes every model's threshold on the balanced `yelp_split.csv` val (no leakage — focal/contrastive exclude it) and also reports threshold-free PR-AUC / ROC-AUC for a fair comparison.

**DistilBERT baseline — details:**
- `distilbert-base-uncased`, MAX_LEN 256, LLRD (decay 0.9) + cosine schedule, 66.96M params; trained on the balanced split (so its test result is directly comparable — table row filled).
- Peaked epoch 3 (val macro F1 69.7% @ 0.48), overfit after (train loss 0.62→0.42 while val F1 declined); early stopping restored epoch 3.
- **Test (threshold 0.48):** Genuine P/R/F1 0.71/0.67/0.69, Fake P/R/F1 0.69/0.72/0.70, accuracy 70%, **macro F1 70%**. Up ~1pp from the old DistilBERT (69%, 128 tok) thanks to the MAX_LEN 256 truncation fix + LLRD/cosine. **Best balanced-test model so far**, edging the baseline LSTM (69%) — consistent with the old "DistilBERT ≥ LSTM by ~1pp".
- **Deployment @13% prior:** fake recall 0.72, FPR 0.33 → fake precision ≈ 25%; per 1,000 reviews flags ~380 (~94 truly fake, ~287 false flags). Same ceiling story as the LSTM.

**DistilBERT focal (γ=2, α=0.878) — details:**
- Same architecture as baseline (`distilbert-base-uncased`, MAX_LEN 256, LLRD + cosine); trained on the imbalanced split with focal loss (loss values ~0.03, expected).
- Peaked epoch 2 (val macro F1 66.6% @ 0.54), declined after (66.4%, 65.9%, 65.1%); early stopping restored epoch 2.
- **Internal imbalanced test (~21% fake — NOT the balanced table):** Genuine P/R/F1 0.86/0.85/0.85, Fake P/R/F1 0.46/0.49/0.48, accuracy 77%, macro F1 66%.
- Essentially unchanged vs the old focal DistilBERT (val 66.3%, Fake F1 0.48) — MAX_LEN 256 barely moved it; ceiling. **Best of the four focal/contrastive variants** on internal test (edges contrastive DistilBERT's 65.8% val / 0.47 Fake F1).
- Projected to balanced 50/50 (fake recall 0.49, FPR 0.15) ≈ 66% macro F1 → still likely below the baseline's 70%.
- **Deployment @13%:** fake recall 0.49, FPR 0.15 → fake precision ≈ 33% — the most conservative of all six (lowest FPR); per 1,000 flags ~194 (~64 truly fake, ~131 false flags).
- **Balanced-test macro F1 + Fake P@13% pending `evaluate_yelp_ablation.py`.** Table row stays "—".

**DistilBERT contrastive (joint SupCon + CE) — details:**
- `distilbert-base-uncased`, **single-stage joint** loss = 0.9·CE + 0.1·SupCon (τ=0.3), MAX_LEN 256, LLRD + cosine, balanced batches; trained on the imbalanced split. 67.2M params (projection head adds ~0.2M over baseline).
- SupCon stayed near its random baseline (4.12 → 3.98; batch=64 baseline log(63)≈4.14) — weak text signal as always — while CE drove the learning. Peaked **epoch 1** (val 65.8% @ 0.64), declined after (64.7%, 64.6%). Training was interrupted mid-epoch-4 by a laptop power loss, but the epoch-1 best was already saved; verified by reloading from disk (val 65.8% @ 0.64) → equivalent to a clean early-stopped run.
- **Internal imbalanced test (~21% fake — NOT the balanced table):** Genuine P/R/F1 0.86/0.83/0.84, Fake P/R/F1 0.44/0.50/0.47, accuracy 76%, macro F1 66%.
- **Big win vs the old staged contrastive DistilBERT** (which collapsed to val 62.3% / Fake F1 0.40 because SupCon-only pretraining wrecked the pretrained encoder): joint training reaches **val 65.8% / Fake F1 0.47** (+3.5pp val). Validates the joint redesign — CE co-supervision every step prevents the encoder damage. Also the **strongest of all four focal/contrastive variants** on internal test (66% vs ~65%).
- Projected to balanced 50/50 (fake recall 0.50, FPR 0.17) ≈ 66% macro F1 → still likely **below** the baseline's 70%.
- **Deployment @13%:** fake recall 0.50, FPR 0.17 → fake precision ≈ 31%; per 1,000 flags ~213 (~65 truly fake, ~148 false flags). More conservative than the baseline (higher precision, lower recall).
- **Balanced-test macro F1 + Fake P@13% pending `evaluate_yelp_ablation.py`** (re-tuned threshold + PR-AUC). Table row stays "—".

**Interpretation:** ~69% is the text-only ceiling on behaviourally-labelled Yelp data; the MAX_LEN/embedding/capacity levers help the model *reach* it (+1pp) but cannot break it, because the labels reflect account behaviour, not text content. Focal and contrastive (LSTM ~65%, DistilBERT ~66% on their imbalanced internal test) handle the minority class but do not beat the balanced baseline — consistent across the old and deduped runs. All six deduped models are now trained; the balanced head-to-head (re-tuned thresholds + PR-AUC) comes from `evaluate_yelp_ablation.py`.

---

### Yelp Fake Detection Models — Final Training Run (clean grouped split, zero leakage)

> **Superseded by the deduped re-run above** — retained for reference (pre-dedup, 252,540-row split, GloVe 100d / MAX_LEN 128–150 models).

**Split:** 252,540 rows (201,900 train / 25,323 val / 25,317 test), grouped by normalized text → 0% train→test leakage.
Previous row-level split had 27.5% train→test leakage and is invalid for reporting.

#### sklearn — TF-IDF (50k features, bigrams, sublinear_tf) + 8 classifiers

KNN trained on 20k subsample only (full 200k too slow). All others trained on full 201,900-row training set.
Thresholds tuned on val set to maximise macro F1; test set never touched until final evaluation.

| Model    | Threshold | Accuracy | Macro F1 | Fake Precision | Fake Recall |
|----------|-----------|----------|----------|----------------|-------------|
| KNN      | 0.81      | 43.9%    | 44.3%    | —              | —           |
| RF       | 0.45      | 66.6%    | 66.4%    | —              | —           |
| DT       | 0.10      | 57.8%    | 57.3%    | —              | —           |
| LR       | 0.46      | 68.2%    | 68.1%    | —              | —           |
| NB       | 0.48      | 66.8%    | 66.8%    | —              | —           |
| XGBoost  | 0.51      | 67.0%    | 67.0%    | —              | —           |
| AdaBoost | 0.50      | 63.4%    | 63.3%    | —              | —           |
| MLP      | 0.13      | 62.3%    | 62.1%    | —              | —           |

**Key observations:**
- LR (68.1%) is the best sklearn model — matches LSTM (68%) and approaches DistilBERT (69%). Linear models suit high-dimensional sparse TF-IDF space.
- LR, XGB, NB, RF cluster at 66–68% — confirming the information ceiling is in the data, not the classifier choice.
- KNN (44.3%) is near-random: the 20k subsample is too small for nearest-neighbor in 50k-dimensional TF-IDF space.
- DT (0.10) and MLP (0.13) have severely miscalibrated probabilities — the models assign very high P(genuine) to fake reviews, and threshold tuning rescues them to 57–62% but cannot overcome the miscalibration.
- All top models have thresholds near 0.5 (0.46–0.51) — well-calibrated, no systematic class bias.

#### LSTM — BiLSTM + GloVe 100d + GlobalMaxPooling1D, SpatialDropout1D(0.3), L2(1e-4), LR=5e-4, EarlyStopping patience=3, ReduceLROnPlateau patience=2

| Epoch | Train Acc | Val Acc | Val Loss | Note |
|-------|-----------|---------|----------|------|
| 1     | 62.1%     | 65.8%   | 0.6412   | |
| 2     | 66.4%     | 67.2%   | 0.6195   | |
| 3     | 67.9%     | 67.2%   | 0.6141   | |
| 4     | 69.3%     | 67.5%   | 0.6116   | Best — saved |
| 5     | 70.8%     | 67.2%   | 0.6187   | |
| 6     | 72.2%     | 66.9%   | 0.6274   | LR halved to 2.5e-4 |
| 7     | 73.7%     | 66.6%   | 0.6309   | Early stop, restored epoch 4 |

Val-tuned threshold: **0.57** (Val macro F1: 67.6%) — threshold optimised for macro F1

Final test results (threshold = 0.57):
| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| Genuine | 0.68      | 0.68   | 0.68 | 12,656  |
| Fake    | 0.68      | 0.68   | 0.68 | 12,661  |

Test Accuracy: 68% | Macro F1: 68%

**Note on earlier run:** A first attempt with the original stacked BiLSTM (random embeddings)
diverged at epoch 1 and reported 63% macro F1 at threshold 0.35. That threshold was wrong —
it was optimising fake-class F1 (sklearn default `average='binary'`), not macro F1. After
fixing the threshold tuning to use `average='macro'`, the correct ceiling is 68%.

**Interpretation:** Model trains healthily across 4 epochs (no catastrophic divergence like
the baseline). Both classes balanced at 68% — threshold 0.57 correctly reflects the model's
slight bias toward genuine. This is the honest LSTM ceiling on behaviourally-labelled Yelp
data with zero leakage and correct macro F1 threshold tuning.

---

### DistilBERT — Yelp fake detection (final clean run, zero leakage)

**Split:** 252,540 rows (201,900 train / 25,323 val / 25,317 test), grouped by normalized text → 0% leakage.
**Checkpoint selection:** threshold-tuned macro F1 on val set after each epoch.

| Epoch | Train Loss | Val Macro F1 | Threshold | Note |
|-------|------------|--------------|-----------|------|
| 1 | 0.615 | 68.5% | 0.52 | Best saved |
| 2 | 0.564 | 69.3% | 0.51 | Best saved ← final checkpoint |
| 3 | 0.509 | 68.6% | 0.51 | Declining |
| 4 | 0.449 | 67.6% | 0.51 | |
| 5 | 0.399 | 67.1% | 0.54 | |
| 6 | 0.362 | 67.1% | 0.51 | |

Val-tuned threshold: **0.51** (Val macro F1: 69.3%)

Final test results (threshold = 0.51):
| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| Genuine | 0.70      | 0.67   | 0.68 | 12,656  |
| Fake    | 0.68      | 0.70   | 0.69 | 12,661  |

Test Accuracy: 69% | Macro F1: 69%

**Interpretation:** Model peaks at epoch 2 and overfits from epoch 3 onward (train loss
continues dropping while val F1 declines). Threshold of 0.51 indicates the model is
well-calibrated with no class bias. DistilBERT outperforms LSTM by 1 percentage point
(69% vs 68%), confirming the ceiling is in the data, not the model architecture.

---

### Focal Loss Ablation — Yelp Fake Detection

**Motivation:** Professor feedback — use focal loss (or contrastive learning) to tackle class imbalance.

**Design:**
- Focal loss: `(1 − p_t)^γ · BCE`, γ=2.0, α=0.878 (genuine fraction → upweights minority fake class)
- Training data: full unbalanced combined dataset from `yelp_split_imbalanced.csv` (~12% fake), instead of the balanced 50/50 `yelp_split.csv`
- Val/test pinned: reviews from `yelp_split.csv` val/test are forced into the same splits here, guaranteeing zero leakage when evaluating on the shared test set
- Architecture: identical to baseline BiLSTM + GloVe 100d — only loss function and training data change

**Data counts (yelp_split_imbalanced.csv after cross-dataset deduplication):**
- Total: ~680k rows (exact count from notebook output)
- Train: ~80% | Val: ~10% | Test: ~10%
- Train fake rate: ~10.4% (natural distribution); Val/test fake rate: ~19–21% (inflated by pinned balanced reviews)

#### LSTM Focal (γ=2.0, α=0.878) — training on imbalanced dataset

| Epoch | Train Acc | Val Acc | Val Loss | Note |
|-------|-----------|---------|----------|------|
| 1     | 65.6%     | 64.2%   | 0.0443   | |
| 2     | 66.9%     | 65.8%   | 0.0431   | |
| 3     | 67.5%     | 64.3%   | 0.0425   | Best — saved |
| 4     | 68.0%     | 64.3%   | 0.0427   | |
| 5     | 68.5%     | 64.8%   | 0.0427   | LR halved to 2.5e-4 |
| 6     | 68.7%     | 66.4%   | 0.0437   | Early stop — restored epoch 3 |

Val-tuned threshold: **0.56** (Val macro F1: 64.7%)

Internal test results (imbalanced test set — ~21% fake, NOT the balanced evaluation set):
| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| Genuine | 0.86      | 0.81   | 0.84 | 53,717  |
| Fake    | 0.42      | 0.50   | 0.45 | 14,157  |

Internal test accuracy: 75% | Internal macro F1: 65%

**Note:** These internal results are on the imbalanced test split and are not directly comparable to baseline results (different class distribution). The official comparison is via `evaluate_yelp_focal.py` on the balanced `yelp_split.csv` test set (25,317 rows, 50/50). That result is pending.

**Observations:**
- Early stopping triggered at epoch 6, restored epoch 3 — only 3 effective training epochs
- Val loss is noisy and the train/val accuracy gap (~4%) is moderate
- Focal loss values (0.03–0.04) are much smaller than baseline BCE values (~0.6) — expected, as focal loss downweights easy examples
- Model still biased toward genuine (fake recall 0.50) even with α=0.878 upweighting

#### DistilBERT Focal (γ=2.0, α=0.878) — training on imbalanced dataset

| Epoch | Train Loss | Val Acc | Val Macro F1 | Threshold | Note |
|-------|------------|---------|--------------|-----------|------|
| 1     | 0.0320     | 76.2%   | 65.7%        | 0.51      | Best saved |
| 2     | 0.0299     | 76.8%   | 66.3%        | 0.53      | Best saved ← final checkpoint |
| 3     | 0.0276     | 76.5%   | 66.0%        | 0.55      | Declining |
| 4     | 0.0247     | 76.6%   | 65.4%        | 0.55      | |
| 5     | 0.0220     | 75.4%   | 64.8%        | 0.53      | |
| 6     | 0.0200     | 75.5%   | 64.6%        | 0.54      | |

Val-tuned threshold: **0.53** (Val macro F1: 66.3%, epoch 2 checkpoint)

Internal test results (imbalanced test set — ~21% fake, NOT the balanced evaluation set):
| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| Genuine | 0.87      | 0.84   | 0.85 | 53,717  |
| Fake    | 0.45      | 0.51   | 0.48 | 14,157  |

Internal test accuracy: 77% | Internal macro F1: 66%

**Observations:**
- Val F1 peaked at epoch 2 (66.3%) and declined monotonically through epochs 3–6 — classic overfitting. Training loss kept falling (0.032 → 0.020) while val F1 declined — the model is memorizing training data past epoch 2.
- Early stopping with patience=2 would have correctly stopped at epoch 4, saving ~2 hours of wasted A100 time.
- Best checkpoint (epoch 2) is correctly saved — the downloaded model is valid.
- Threshold of 0.53 is close to 0.5 — well-calibrated, no strong class bias.
- Both focal models (LSTM: 65%, DistilBERT: 66%) score lower internally than baseline on the imbalanced test set, but the internal test set has a different class distribution — the official comparison via `evaluate_yelp_focal.py` on the balanced test set is still pending.

---

### Contrastive Learning Ablation — Yelp Fake Detection

**Motivation:** Professor feedback — use focal loss or contrastive learning to tackle class imbalance.

**Method:** Two-stage supervised contrastive learning (SupCon, Khosla et al. 2020)
- Stage 1: encoder + projection head trained with SupCon loss on class-balanced batches (128 genuine + 128 fake). Temperature=0.07. 7% word dropout augmentation.
- Stage 2: projection head discarded; classifier head (Dropout(0.5) → Dense(1, sigmoid)) attached to encoder and trained on the full imbalanced set with class-weighted BCE.
- Same `yelp_split_imbalanced.csv` training data as focal models (~12% fake).

**Architecture (LSTM contrastive):**
- Encoder: Embedding(GloVe 100d) → SpatialDropout1D(0.3) → BiLSTM(128) → GlobalMaxPooling1D → 256-dim output
- Projection head (Stage 1 only): 256 → Dense(128, relu) → Dense(64) → L2-normalize → 64-dim unit sphere
- Classifier (Stage 2): 256 → Dropout(0.5) → Dense(1, sigmoid)

#### LSTM Contrastive — training results (v2, improved hyperparameters)

**Improvements over initial run:** temperature 0.07→0.15, GloVe frozen in Stage 1 / unfrozen in Stage 2, Stage 1 epochs 5→10, projection head 256→128→64 widened to 256→256→128, Stage 2 LR 5e-4→2e-4, Stage 2 balanced batches (50/50) replacing extreme class weights (4.47×).

**Stage 1 — SupCon pretraining (10 epochs, batch=256, LR=1e-3, temperature=0.15, GloVe frozen):**

| Epoch | SupCon Loss |
|-------|-------------|
| 1     | 5.5334      |
| 2     | 5.5283      |
| 3     | 5.5257      |
| 4     | 5.5241      |
| 5     | 5.5220      |
| 6     | 5.5205      |
| 7     | 5.5187      |
| 8     | 5.5204      |
| 9     | 5.5162      |
| 10    | 5.5148      |

**Stage 2 — Classifier fine-tuning (balanced batches, LR=2e-4, patience=3 on val macro F1):**

| Epoch | Train Acc | Val Acc | Val Loss | Val Macro F1 | Threshold | Note |
|-------|-----------|---------|----------|--------------|-----------|------|
| 1     | 72.6%     | 64.5%   | 0.6238   | 64.65%       | 0.66      | Best saved ← final checkpoint |
| 2     | 73.3%     | 67.8%   | 0.5912   | 64.58%       | 0.63      | patience 1/3 |
| 3     | 74.0%     | 66.8%   | 0.6071   | 64.19%       | 0.66      | patience 2/3, LR 2e-4→1e-4 |
| 4     | 74.6%     | 66.1%   | 0.6157   | 64.21%       | 0.67      | patience 3/3 — early stop |

Val-tuned threshold: **0.66** (Val macro F1: 64.7%)

Internal test results (imbalanced test set — ~21% fake, NOT the balanced evaluation set):
| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| Genuine | 0.86      | 0.82   | 0.84 | 53,717  |
| Fake    | 0.42      | 0.50   | 0.45 | 14,157  |

Internal test accuracy: 75% | Internal macro F1: 65%

**Improvement vs initial contrastive run:** val macro F1 63.6% → **64.7%** (+1.1pp); Fake F1 0.43 → 0.45.

**Observations:**
- SupCon loss decreased only 5.533 → 5.515 over 10 epochs — smaller absolute drop than the first run (5.529 → 5.453 over 5 epochs with temp=0.07). Higher temperature + frozen embeddings both reduce gradient magnitude, so the absolute loss units are not directly comparable. The loss is still near the random-initialization ceiling (~log(255) ≈ 5.54), confirming that the BiLSTM can extract only marginal discriminative signal from GloVe embeddings of Yelp reviews.
- Stage 2 started at 72.6% train accuracy vs 70.4% in the first run — higher starting point indicates the frozen-embedding Stage 1 produced better LSTM representations despite the smaller loss drop.
- Stage 2 peaked at epoch 1 (64.65%) and declined monotonically — same rapid overfitting pattern.
- Early stopping (patience=3) with LR reduction triggered correctly at epoch 4.
- Contrastive LSTM (64.7%) now **ties focal LSTM** (64.7%) but both remain below baseline LSTM (67.6%). Most of the gain vs the first contrastive run came from Stage 2 improvements (balanced batches + lower LR), not from better SupCon representations.
- The improvement from Stage 2 changes (class-weighted → balanced batches) aligns with the hypothesis that balanced training matches the evaluation objective (50/50 test set) more directly.

**Conclusion on SupCon for LSTM on Yelp:**
Yelp fake labels are behavioral (account patterns), not textual. Fake and genuine reviews occupy nearly identical text-feature space. SupCon pushing fake embeddings away from genuine on the unit sphere has almost no signal to exploit, which is why the loss barely moves from random. The encoder is constrained by the data, not the method. Contrastive pretraining provides a minor benefit over focal loss (they tie at 64.7%) but neither approach recovers the 3pp gap vs the balanced baseline (67.6%). This is a meaningful negative result: for behaviorally-labeled data, the text-only signal ceiling is fixed regardless of training strategy.

**Official comparison** (on balanced `yelp_split.csv` test set) pending from `evaluate_yelp_ablation.py`.

#### DistilBERT Contrastive — training results

**Architecture:**
- Encoder: `distilbert-base-uncased` → CLS token → 768-dim
- Projection head (Stage 1 only): 768 → Dense(256, relu) → Dense(128) → L2-normalize → 128-dim unit sphere
- Classifier head (Stage 2): pre_classifier(768→768, relu) → Dropout(0.1) → Dense(768→2)

**Three-stage training:**
- Stage 1: SupCon on balanced batches (32 genuine + 32 fake, batch=64), temperature=0.07, LR=2e-5, 5 epochs
- Stage 2a: Frozen encoder, classifier head warmup, 1 epoch, LR=1e-4
- Stage 2b: Full fine-tuning, LR=1e-5, patience=3 on val macro F1, up to 5 epochs

**Stage 1 — SupCon pretraining (random baseline for N=64: log(63) ≈ 4.143):**

| Epoch | SupCon Loss |
|-------|-------------|
| 1     | 4.1013      |
| 2     | 4.0013      |
| 3     | 3.8878      |
| 4     | 3.7932      |
| 5     | 3.7233      |

Loss dropped from 4.10 → 3.72 — a 10% drop below the random-initialization baseline (4.14). This is more absolute movement than LSTM contrastive (which barely left log(255)≈5.54), but still modest.

**Stage 2a — Frozen encoder warmup (1 epoch):**
- Train Loss: 0.1395 | Val Macro F1: **62.3%** | Threshold: 0.26

**Stage 2b — Full fine-tuning (early stopping at patience=3):**

| Epoch | Train Loss | Val Macro F1 | Threshold | Note |
|-------|------------|--------------|-----------|------|
| 1     | 0.1229     | 62.1%        | 0.10      | No improvement (1/3) |
| 2     | 0.0980     | 61.8%        | 0.10      | No improvement (2/3) |
| 3     | 0.0760     | 61.4%        | 0.11      | Early stop — best = Stage 2a |

Final checkpoint: **Stage 2a** (Val Macro F1 = 62.3%, threshold = 0.26)

Internal test results (imbalanced test set — ~21% fake):
| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| Genuine | 0.84      | 0.84   | 0.84 | 53,717  |
| Fake    | 0.40      | 0.41   | 0.40 | 14,157  |

Internal test accuracy: 75% | Internal macro F1: 62%

**Comparison to baseline DistilBERT (internal val, same dataset):**
- Baseline: Val Macro F1 = 69.3%, threshold ≈ 0.51
- Contrastive: Val Macro F1 = 62.3%, threshold = 0.26
- **DistilBERT contrastive is 7pp worse than the baseline.**

**Observations:**
- Stage 2b never improved over Stage 2a — every fine-tuning epoch degraded val F1 monotonically. This is distinct from LSTM contrastive where Stage 2 at least recovered somewhat. Here, unlocking the encoder immediately caused catastrophic forgetting.
- Threshold of 0.26 is a major red flag: the model outputs very low fake probabilities, indicating the classifier head is poorly calibrated. The SupCon representations do not produce a clear fake/genuine separation in logit space.
- Stage 2b thresholds collapsed to 0.10 — the model is effectively predicting near-zero probability for fake on almost all reviews. The full fine-tuning is destroying the weak structure built in Stage 2a.
- SupCon loss moved more for DistilBERT (10% below random baseline) than LSTM (~0.3% below), but the larger movement may actually be the problem: DistilBERT's rich pretrained representations were more successfully reshaped by SupCon, but this reshaping pointed them in a direction that is useless for fake/genuine classification.

**Conclusion on SupCon for DistilBERT on Yelp:**
Contrastive pretraining harmed DistilBERT more than it harmed LSTM. The root cause is the same — behavioral Yelp labels give no text-level discriminative signal — but the consequence is more severe here because SupCon overwrote the rich pretrained representations that DistilBERT already had. The baseline DistilBERT achieved 69.3% val macro F1 by starting from powerful pretrained weights and fine-tuning with BCE; the contrastive version spent 5 epochs distorting those weights on a label signal that is orthogonal to text content, and recovered only 62.3%. The behavioral-label ceiling is fixed, but SupCon pretraining actively degraded what the model could recover in Stage 2.

**Official comparison** (on balanced `yelp_split.csv` test set) pending from `evaluate_yelp_ablation.py`.

---

### DistilBERT — Yelp fake detection (preliminary runs, pre-leakage-fix — for reference only)

**Note:** Runs 1–4 below used a row-level split with 27.5% train→test leakage.
Results are inflated and not comparable to final sklearn/LSTM/DistilBERT results.
Retained for context on experimental progression only.

**Run 1** — full YelpZip + YelpChi (~676k reviews), LR=2e-5, weighted cross-entropy loss
| Epoch | Loss   | Val Accuracy | Val F1 |
|-------|--------|--------------|--------|
| 1     | 0.6061 | 70.3%        | 37.3%  |
| 2     | 0.5590 | 70.7%        | 37.9%  |
| 3     | 0.5021 | 74.7%        | 38.4%  |

**Run 2** — balanced ~178k reviews (all fake + equal genuine sample), LR=1e-5, plain cross-entropy, early stopping (patience=2)
| Epoch | Loss   | Val Accuracy | Val F1 | Note |
|-------|--------|--------------|--------|------|
| 1     | 0.6195 | 69.2%        | 71.8%  | Best model saved |
| 2     | 0.5771 | 69.5%        | 70.4%  | No improvement (1/2) |
| 3     | 0.5418 | 69.8%        | 70.6%  | Early stop |

**Run 3** — balanced ~252k reviews (YelpZip + YelpChi + YelpNYC), LR=1e-5, no early stopping, 6 epochs
| Epoch | Loss   | Val Accuracy | Val F1 | Note |
|-------|--------|--------------|--------|------|
| 1     | 0.6131 | 68.0%        | 70.6%  | |
| 2     | 0.5641 | 70.4%        | 70.1%  | |
| 3     | 0.5142 | 71.2%        | 71.9%  | |
| 4     | 0.4582 | 71.6%        | 74.4%  | Best |
| 5     | 0.4101 | 72.8%        | 74.0%  | |
| 6     | 0.3741 | 72.9%        | 74.7%  | Best |

**Run 4** — continued from Run 3 best checkpoint, fresh optimizer (no warmup), 4 more epochs
| Epoch | Loss   | Val Accuracy | Val F1 | Note |
|-------|--------|--------------|--------|------|
| 1     | 0.4003 | 73.2%        | 75.9%  | |
| 2     | 0.3286 | 75.0%        | 77.0%  | |
| 3     | 0.2694 | 76.2%        | 77.6%  | |
| 4     | 0.2293 | 76.5%        | 77.9%  | Best |

Final test results at threshold 0.50 (best model = Run 4 epoch 4):
- Genuine: precision 0.80, recall 0.70, F1 0.75
- Fake:    precision 0.73, recall 0.83, F1 0.78
- Accuracy: 76.5% | Macro F1: 76%

Threshold tuning: best threshold 0.30 → Fake F1 78.3% (genuine recall drops to 64%, macro F1 drops to 75% — not worth it, use 0.50)

---

## Key Findings

### Balanced sampling dramatically improved DistilBERT F1 on Yelp
Run 1 (full imbalanced 676k dataset, weighted loss) achieved only 38.4% F1. Switching
to balanced 1:1 sampling nearly doubled it to 71.8%. Further training with more data
(adding YelpNYC, 252k balanced) and continued training across 10 total epochs brought
the final model to 77.9% F1 — more than double the original result.

The key interventions in order of impact:
1. Balanced sampling (1:1 ratio) — biggest single improvement
2. Adding YelpNYC to training data (+41% fake examples)
3. Removing early stopping — F1 continued improving past epoch 6 up to epoch 10
   *(observation from the leaky preliminary runs only — see correction note below)*

### Early stopping with low patience can be harmful
With patience=2, training would have stopped at epoch 3 (F1=71.9%) and missed the
continued improvements at epochs 4-10 (final F1=77.9%). The F1 oscillated between
epochs before recovering — patience=2 was too aggressive. For datasets with weak
signal, F1 may temporarily dip before continuing to improve.

**Correction:** The "removing early stopping" finding above is based on the leaky
preliminary runs (Runs 1–4, 27.5% train→test leakage). In the clean final run
(grouped split, zero leakage), DistilBERT peaked at epoch 2 (F1=77.6%) and overfit
monotonically after that — the opposite pattern. Early stopping at patience=2 would
have been correct in the clean run. The finding is valid only as a description of
experimental progression under the leaky split, not as a general training recommendation.

### Text-only models converge to the same ceiling regardless of architecture
All ten Yelp fake detection models cluster between 66% and 69% macro F1:
LR (68.1%), LSTM (68%), DistilBERT (69%), XGB (67%), NB (66.8%), RF (66.4%).
The span from simplest (NB) to most complex (DistilBERT) is only 2.3 percentage points.
This is strong evidence that the ceiling is in the data labeling, not the model.

**Why:** Yelp's spam filter assigns labels based on behavioral signals —
account age, posting frequency, reviewing patterns — not based on how the
text reads. Paid reviewers and sellers posing as customers write fluent,
natural text that is indistinguishable from genuine reviews at the word
level. This means text-only models have no reliable signal to learn from.

**Implication for the thesis:** This finding directly supports the argument
that different types of fake reviews require different detection approaches.
Text-based models (TF-IDF, LSTM, DistilBERT) are effective for
computer-generated fakes (CG/OR dataset) but approach a ceiling for
behaviorally-suspicious fakes (Yelp dataset). The dashboard's behavioral
analysis — duplicate texts, same-day posting — is the appropriate tool for
the latter category.

### Logistic Regression matches deep learning on TF-IDF features
LR (68.1%) matches LSTM (68%) and is within 1% of DistilBERT (69%) on the Yelp task,
despite being orders of magnitude simpler. This confirms that TF-IDF features already
capture most of the available text signal — the remaining 1% DistilBERT advantage comes
from contextual understanding, but it cannot recover the signal missing from the labels
themselves. For high-dimensional sparse text features, LR is the most efficient choice.

### Balanced training requires threshold recalibration for real-world deployment
The model was trained and evaluated on 50/50 balanced data. In the real world,
only ~13% of Yelp reviews are fake. A model calibrated for 50/50 will over-predict
fake on naturally distributed data, generating excessive false positives. For
frontend deployment, the classification threshold should be raised above 0.50
(e.g. 0.65-0.70) so the predicted fake rate matches the known ~13% real-world rate.

### Row-level splitting on multi-dataset corpora causes severe leakage
The initial random row-level 80/10/10 split produced 27.5% train→test leakage —
meaning 27.5% of test rows had identical (or near-identical) text already seen
during training. Root cause: the same review text appears in multiple Yelp datasets
(YelpZip, YelpNYC share spam campaigns; YelpChi overlaps with both). Of 252,540
balanced rows, 43,187 (17.1%) were exact duplicates across datasets.

Fix: grouped splitting at the normalized-text level. All occurrences of a text
(after lowercase + whitespace collapse) are assigned to the same split. After this
fix, train→val and train→test leakage dropped to exactly 0%.

This leakage would have inflated all model metrics, especially LSTM and DistilBERT
which can memorize specific review texts. The 27.5% figure is high enough to
completely invalidate any results from the row-level split.

### SMOTE is not appropriate for this dataset
SMOTE (Synthetic Minority Over-sampling Technique) was considered but not applied.
For Yelp, the fake and genuine classes overlap heavily in text space because labels
are behavioral, not textual. SMOTE interpolates between existing minority-class
feature vectors — on this data, those interpolations land in the same overlapping
region as genuine reviews and add no new discriminative signal. Balanced sampling
of real examples is more appropriate. SMOTE is effective when the minority class
occupies a distinct region in feature space (e.g. credit card fraud, medical
diagnosis), which is not the case here.

---

## Yelp Frontend Dataset

The frontend demonstration uses only the held-out test split, cross-referenced against
the raw dataset files to recover metadata (business ID, rating, date, user ID).

### Final counts (after deduplication)
- **20,923 unique reviews** — 8,922 fake (42.6%), 12,001 genuine (57.4%)
- Across **4,044 businesses** and **19,455 unique users**
- Sources: YelpZip (19,154), YelpNYC (96), YelpChi restaurants (1,539), YelpChi hotels (134)

### Why 20,923, not 25,317
The test split in `yelp_split.csv` has 25,317 rows, but some review texts appear more than
once — the same text was collected independently into multiple raw datasets (e.g. a NYC
review that appears in both YelpZip and YelpNYC). The grouped split correctly assigns all
occurrences of a text to the same split, so these duplicates all land in the test split
together. After deduplication by normalized text, 20,923 unique reviews remain.

### Cross-dataset duplicates: 33,035 → 20,923
A naive cross-reference (scanning all four raw files without deduplication) produced
33,035 entries — 12,112 more than the true test split size. Root cause: YelpNYC is
geographically a subset of YelpZip (both cover NYC businesses). Reviews from NYC
businesses appear verbatim in both files. After fixing `prepare_yelp_frontend.py` to
track seen normalized texts and skip duplicates, the count correctly matches 20,923.

### Note on test split vs. training
All 20,923 frontend reviews are from the test split — they were never seen during
training or validation. The train/val/test split was performed at the grouped-text
level (all occurrences of a text go to the same split), guaranteeing zero leakage.

---

## Human-Written Fake Review Detection (Ott + Li 2014)

A **third, distinct fake-detection axis**: the project already covers **AI/computer-generated** fakes (Model A, CG/OR) and **behavioural/relational** spam (Yelp text + GNN); this adds **human-written deception** — reviews that real people were paid to fabricate, detected from text alone. This is the classic *deceptive opinion spam* task, and it's the hardest "human fake" signal (the text is genuinely human, just dishonest).

**Dataset (`build_human_fake_dataset.py` → `data/human_fake/human_fake_reviews.csv`):** the two gold-standard corpora combined — **Ott** Deceptive Opinion Spam v1.4 (1,600 hotel reviews; deceptive = MTurk, truthful = TripAdvisor/Web) + **Li et al. 2014** cross-domain (hotel / restaurant / **doctor**; deceptive = MTurk + domain experts). After exact-dedup (Li's hotel reuses Ott's hotel reviews) and binary-junk removal: **2,832 reviews — 1,636 fake / 1,196 genuine**, across hotel (1,876) / doctor (556) / restaurant (400). Stratified 80/10/10 (train 2,265 / val 283 / test 284). **Design note:** unlike the CG/OR and sentiment models, this **KEEPS English stop words** — function-word / personal-pronoun usage is a known deception cue (Ott et al. 2011).

**Results (clean test, n=284; macro-F1):**

| Model | macro-F1 | fake recall | fake precision |
|---|---|---|---|
| **Logistic Regression** (sklearn) | **88.2%** | 95.7% | 86.3% |
| MLP | 87.5% | 86.6% | 91.6% |
| **DistilBERT** (fine-tuned, Colab) | **87.6%** | 91.5% | 88.2% |
| Naïve Bayes | 86.4% | 92.7% | 85.9% |
| Random Forest | 86.1% | 92.1% | 85.8% |
| sklearn ensemble (majority) | 86.6% | 89.0% | 88.5% |
| XGBoost | 82.4% | · | · |
| KNN / AdaBoost / DT | 77.8 / 75.0 / 66.0% | · | · |

**Findings:** ~88% — **in line with the published Ott benchmark**. Notably **DistilBERT (87.6%) ≈ Logistic Regression (88.2%)**: on a ~2.3k-row corpus the deep model has no advantage over a TF-IDF linear model (small data favours simpler models; the transformer was clean — early-stopped at its epoch-4 val peak, no overfit). 8 sklearn models → `data/human_fake/human_fake_sklearn_models/`; DistilBERT → `data/human_fake/human_fake_distilbert/` (`train_human_fake_distilbert.ipynb`, Colab GPU).

**⚠️ Honest caveat (for the defense):** these are gold labels but the fakes are **MTurk-*elicited*** (people *asked* to write fake reviews), a controlled proxy — Turkers write detectably differently from real reviewers, which is partly why ~88% is achievable. So report this as the **classic deceptive-opinion-spam benchmark**, NOT as real-world paid-spam performance; cross-domain/real-world transfer is weaker (the known limitation of this subfield). It complements — does not replace — the behavioural Yelp signal (real-world human fakes, but only ~69% from text because they're textually indistinguishable).

---

## Unified Fake Detector — CG **and** human deception in one model

Combines the two fake types into a **single detector** that flags a review fake if it is
EITHER computer-generated OR human-deceptive (`build_combined_fake_dataset.py` →
`data/combined_fake_reviews.csv`, **131,768 rows**: 64,478 cg_fake + 1,636 human_fake +
65,654 genuine; CG/OR splits + Ott/Li splits preserved → no leakage). Rows are tagged
`fake_type ∈ {cg_fake, human_fake, genuine}` for per-type reporting. **Pre-flight shortcut
check:** the human domains (hotel/doctor/restaurant) each contain *both* fake and genuine
examples, so the model can't shortcut on topic. Human-origin TRAIN rows are **oversampled
15×** so the 2.5%-minority human signal is learned (val/test untouched). Stop words KEPT.

**Headline = per-fake-type recall on the clean combined test (n=13,179; human-fake ≈164):**

**Exact full table** (`compute_unified_dl_preds_colab.ipynb` GPU DL → `unified_dl_preds.csv`; `merge_unified_ensemble.py` local sklearn + ensembles; all per-model + the 10-model ensemble row):

| model | macro-F1 | CG-fake recall | human-fake recall | genuine spec |
|---|---|---|---|---|
| **DistilBERT** (fine-tuned, ONNX) | **98.7%** | 99.5% | **92.7%** | 98.1% |
| **LSTM** (BiLSTM+GloVe, ONNX) | 97.2% | 97.6% | 84.8% | 97.2% |
| **MLP** (sklearn) | 95.6% | 96.6% | 82.3% | 95.1% |
| Logistic Regression | 94.7% | 95.1% | 81.1% | 94.7% |
| XGBoost | 92.1% | 90.4% | 70.7% | 94.2% |
| KNN | 90.0% | 87.9% | 63.4% | 92.7% |
| Random Forest | 89.0% | 86.0% | 67.1% | 92.4% |
| Naïve Bayes | 88.7% | 84.7% | 62.8% | 93.4% |
| Decision Tree | 79.9% | 80.5% | 46.3% | 80.1% |
| AdaBoost | 77.4% | 66.3% | 36.0% | 90.0% |
| **sklearn majority (8-vote)** | 92.7% | 88.7% | 64.6% | 97.3% |
| **FULL 10-model ensemble** (served by Analyze) | **95.1%** | 92.7% | **73.8%** | 98.0% |

**10-model ensemble note (filled in 2026-06):** adding the two strong DL members to the 8-sklearn vote lifts the ensemble — macro-F1 92.7%→95.1%, and **human-fake recall 64.6%→73.8% (+9.2 pp)** — but it **still trails the best single model** (DistilBERT 92.7% human recall): the 8 sklearn (several weak on human fakes) outvote the 2 deep models on the scarce human class. Genuine specificity stays high (98.0%). Confirms (c): for the rare human-deception class, **best-single-model (DistilBERT) > majority vote**, which is why Analyze surfaces per-model dots, not just a consensus.

**Human-fake recall is on only n=164 test reviews — report with 95% Wilson CIs (`human_fake_recall_ci.py`):** DistilBERT 92.7% [87.6–95.8] (152/164) · LSTM 84.8% [78.5–89.5] · MLP 82.3% [75.8–87.4] · LR 81.1% [74.4–86.4] · XGB 70.7% [63.4–77.2] · RF 67.1% [59.6–73.8] · KNN 63.4% [55.8–70.4] · NB 62.8% [55.2–69.8] · DT 46.3% [38.9–54.0] · AdaBoost 36.0% [29.0–43.6] · 8-sklearn-majority 64.6% [57.1–71.5] · **10-model ensemble 73.8% [66.6–79.9]**. CIs are ±4–7 pp (the n=164 limitation made explicit). **Statistically:** DistilBERT's CI floor (87.6%) clears almost every other point estimate → its lead is real; but mid-tier models (MLP/LR, KNN/RF/NB) have overlapping CIs → indistinguishable at this n; and the 10-model ensemble's CI [66.6–79.9] does **not** overlap DistilBERT's [87.6–95.8] → ensemble is *significantly* below the best single model, not just nominally. Always present human-fake recall as "X% [CI], k/164". Same MTurk-elicited validity caveat as the standalone human detector.

**Findings:** (a) **one model genuinely catches both** — **DistilBERT is best (98.7% macro-F1; CG 99.5% / human 92.7% / genuine 98.1%)**; LSTM, MLP, LR also catch both (CG 95–98% / human 81–84%). The oversampling prevented collapse to a CG-only detector. **DistilBERT's pretraining gives the biggest lift exactly on the hard, scarce human-deception signal (92.7% vs ~83% for the from-scratch LSTM / TF-IDF models)** — the one place transfer learning matters most. (b) The CG>human asymmetry is expected — human deception is the harder, scarcer signal even after oversampling (cf. the standalone human-only detector at ~88% on a balanced human test). (c) **Weak learners (DT/AdaBoost) fail the minority human signal (36–46%)** and **the naive majority ensemble's human recall (64.6%) is *worse* than the best single model (MLP 82.3%)** — the weak voters drag it down, so for this task serve **MLP/LR/LSTM**, not majority vote. (d) macro-F1 is CG-weighted (CG dominates the test); **human-fake recall is the informative per-type number** (on ~164 human test rows; same MTurk-elicited caveat as the standalone detector).

**Artifacts:** `train_combined_fake_sklearn.py` → `data/combined_fake_models/` (8 sklearn); `train_combined_fake_lstm.ipynb` → `data/combined_fake_lstm_onnx/`; `train_combined_fake_distilbert.ipynb` → `data/combined_fake_distilbert/`.

**SERVED EVERYWHERE — fully replaces Model A in the live app.** The unified detector now supplies fake verdicts for **all three** live fake-detection paths: the Fake Review Detection browse cards (`GET /reviews?task=fake`), the user-history chart (`score_fake_distilbert`), and the Dashboard Analyze box (`/analyze` → `_merge_fake_sentiment`). It's the *true* generalized fake detector — catches both AI-generated **and** human-deceptive reviews, where Model A was AI/CG-only — and is a strict **superset** of Model A (same `fake_reviews_broadened.csv` CG/OR data + the Ott/Li human-deception corpus). New module `model_unified.py` (same `fake_models` + phrase/sentence-highlight output shape as `model_a.py`) loads `data/combined_fake_models/` (8 sklearn) + `data/combined_fake_lstm_onnx/` (BiLSTM) + **`data/combined_fake_distilbert_onnx/`** — the HF checkpoint exported to ONNX by `convert_combined_distilbert_to_onnx.py` (torch↔onnx logits match to 1e-6) so it serves via onnxruntime with no torch/TF at runtime. Preloaded at startup. **`model_a.py` is retired from serving** (no longer imported/loaded by `main.py`); it + the `data/fake_broadened_*` models stay on disk only for the offline comparison scripts (`amazon_detector_comparison.py`, `yelp_labeled_human_fake_test.py`).

**Frontend demo — labeled Amazon browse set (ground-truth view).** The Fake Review Detection Amazon panel defaults to a **labeled held-out set** (`build_amazon_fake_labeled.py` → `data/amazon_fake_labeled.csv`, the CG/OR products test split with 1=fake/0=genuine; served by `load_amazon_labeled.py` via `GET /amazon-fake-labeled`), so the demo can show predicted-vs-true side by side (predicted-fake / predicted-real by 10-model majority + per-class correctness), mirroring the Yelp section. The second dropdown option is the original live **Amazon Fine Food** browse-by-Product-ID path (unlabeled). An earlier third option, the **HC3 out-of-distribution** probe, was removed from the UI (the number stays in §1b above; backend `dataset='ood'` is retained for offline use).

**Calibration check before serving** (`model_unified.score_fake` on 400 random real Amazon reviews, pre-LLM → effectively all genuine): sklearn-majority flags **1.0%**, LSTM **2.8%**, DistilBERT **1.8%**, and the **full 10-model consensus only 0.2%** — i.e. the generalized detector does **not** over-flag genuine uploads (the standalone human-deception detector, by contrast, fired ~38% here — weak cross-domain transfer). Confirms the unified model stayed well-calibrated on real reviews while gaining human-deception coverage. *(Note: short/atypical hand-written text can still draw a high sklearn vote — e.g. a 1-line generic review hit 7/10 sklearn — but the DL models and the real-data consensus stay correct.)*

### Ablation — can we also fold Yelp into the unified text detector? (No.)

*Motivation:* if CG + Ott/Li human deception combine into one text model, why not add YelpZip+NYC+Chi too, for a single all-fakes detector? **Because Yelp's label is behavioral, not textual** — the project's central finding. Tested directly (`train_unified_plus_yelp_sklearn.py`; same lr/nb/mlp/xgb + majority trained two ways: **A** = CG+human only, **B** = + a balanced 60k Yelp-text sample; evaluated on the combined test *and* the yelp_split test, per fake type):

| variant | CG-R | human-R | **genuine-spec** | Yelp-fake-R | Yelp macro-F1 |
|---|---|---|---|---|---|
| **A** unified (no Yelp) | 91.8% | 71.3% | **97.0%** | 3.4% | 36.6% |
| **B** unified + Yelp text | 95.8% | 89.6% | **86.3%** | 43.8% | 59.9% (best single 64.3%) |

**Findings (more nuanced than "text is pure noise on Yelp", but the same conclusion):**
1. **The 3-way text model is the *worst* Yelp detector.** Even trained on Yelp, B reaches only **60% Yelp macro-F1** (best single 64%) — *below* the dedicated **Yelp text-only ceiling (69%)**, far below **Model B text+behavioral (76–77%)** and the GNN. Mixing CG/human in *dilutes* Yelp-from-text below even a Yelp-only text model. So there is a *weak* textual correlate (~60–69%, not chance), but it's nowhere near the right-modality approaches.
2. **The CG/human recall "gains" are a threshold artifact, not capability.** They come with a **−10.6 pp collapse in genuine specificity (97% → 86%)** — B simply flags more of *everything*. Recall↑ because specificity↓.
3. **It would damage the served Analyze box:** genuine specificity 97%→86% ≈ **~14% of genuine uploads flagged fake** (vs ~3% now). A clear regression for the deployed detector.

**Conclusion:** confirmed — keep the **modality separation**. The unified text detector handles textually-detectable fakes (CG + human deception); Yelp behavioral spam belongs to **Model B** (behavioral) + the **GNN** (relational). Folding Yelp into the text model produces a mediocre Yelp detector *and* over-flags genuine reviews. (Note: A's 36.6% Yelp F1 is just an off-domain model applied cold — the meaningful Yelp baselines are the 69% dedicated text model and 76–77% Model B, both of which B fails to reach.)

---

## Master Dataset Citations (thesis references section)

Every dataset the project has trained or evaluated on, with its citation. The two detailed
tables further down (review/non-review auxiliary sources; CG/OR augmentation sources) expand
the rows marked "→ see table". **Best-effort — verify the exact citation format + current
license on each source page before publishing.**

| # | Dataset (file / where used) | Citation | Link | License |
|---|---|---|---|---|
| 1 | **Amazon Fine Food Reviews** (`data/Reviews.csv`; live Amazon section, sentiment eval, review-detector positives, CG/OR "real" reviews) | McAuley & Leskovec, "From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews," **WWW 2013** | https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews (orig. Stanford SNAP) | research use |
| 2 | **Fake Reviews Dataset (CG/OR)** (`data/fake reviews dataset.csv`; Amazon fake-detection training, CG/OR detector) | Salminen, Kandpal, Kamel, Jung & Jansen, "Creating and detecting fake reviews of online products," **J. Retailing and Consumer Services 64 (2022)** | https://osf.io/tyue9/ (also Kaggle "Fake Reviews Dataset") | research use |
| 2b | **Human-deception corpus** (`data/human_fake/`; the human-written fake-review detector) | **Ott, Choi, Cardie & Hancock, "Finding Deceptive Opinion Spam by Any Stretch of the Imagination," ACL 2011** (positive) + **Ott, Cardie & Hancock, "Negative Deceptive Opinion Spam," NAACL 2013** (negative) + **Li, Ott, Cardie & Hovy, "Towards a General Rule for Identifying Deceptive Opinion Spam," ACL 2014** (cross-domain) | https://myleott.com/op-spam.html · http://cs.stanford.edu/~bdlijiwei/deception_dataset.zip | research use |
| 3 | **IMDB 50K Movie Reviews** (`data/IMDB Dataset.csv`; sentiment training, review-detector positives) | Maas, Daly, Pham, Huang, Ng & Potts, "Learning Word Vectors for Sentiment Analysis," **ACL 2011** | http://ai.stanford.edu/~amaas/data/sentiment/ | research use |
| 4 | **Sentiment140** (`data/tweets.csv`, 1.6M tweets; sentiment training) | Go, Bhayani & Huang, "Twitter Sentiment Classification using Distant Supervision," **Stanford CS224N project report, 2009** | http://help.sentiment140.com/ | research use |
| 5 | **YelpZip + YelpNYC** (raw; Yelp fake-detection, multi-modal, GNN — `data/yelp_split.csv`) | Rayana & Akoglu, "Collective Opinion Spam Detection: Bridging Review Networks and Metadata," **KDD 2015** | http://shebuti.com/collective-opinion-spam-detection/ | research use (academic) |
| 6 | **YelpChi** (restaurants + hotels; part of the Yelp graph/GNN) | Mukherjee, Venkataraman, Liu & Glance, "What Yelp Fake Review Filter Might Be Doing?," **ICWSM 2013** (also redistributed by Rayana & Akoglu, KDD 2015) | http://shebuti.com/collective-opinion-spam-detection/ | research use (academic) |
| 7 | **Amazon Musical Instruments fraud graph** (`Amazon.mat`; the 2nd GNN dataset) | **Data:** McAuley & Leskovec, WWW 2013. **25 node features:** Liu, Ao, Qin, Chi, Feng, Yang & He, "Alleviating the Inconsistency Problem of Applying GNN to Fraud Detection," **SIGIR 2020** (arXiv:2005.10150). **Graph + fraud labels (as used):** Dou, Liu, Sun, Deng, Peng & Yu, "Enhancing GNN-based Fraud Detectors against Camouflaged Fraudsters" (CARE-GNN), **CIKM 2020** (arXiv:2008.08692) | https://github.com/YingtongDou/CARE-GNN (also DGL `FraudAmazonDataset`) | research use (academic) |
| 8 | **HC3 (Human ChatGPT Comparison Corpus)** (out-of-distribution eval of the CG/OR detector) | Guo, Zhang, Wang, Jiang, Nie, Ding, Yue & Wu, "How Close is ChatGPT to Human Experts? Comparison Corpus and Detection," **2023** (arXiv:2301.07597) | https://huggingface.co/datasets/Hello-SimpleAI/HC3 | CC BY-SA / per card |
| 9 | **Review/Non-Review auxiliary sources** (Steam, Amazon-Reviews-2023, app_reviews, Google-Local, CMU Movies, Amazon-QA, SQuAD, WikiText, EmpatheticDialogues, AESLC, LexGLUE, CodeSearchNet, WritingPrompts, …) | → see **"Review / Non-Review Detection — Dataset Sources"** table below | (per row) | (per row) |
| 10 | **CG/OR augmentation / AI-generator sources** (Lyra AI, sutro, Kenshiii, + open-model generations: Qwen, Llama-3, Zephyr, Yi; and GPT-4o-mini) | → see **"Broadened Fake-Review (CG/OR) Detection — Dataset Sources"** table below | (per row) | (per row) |

**Methods/models also worth citing** (not datasets): GraphSAGE (Hamilton et al., NeurIPS 2017), GAT (Veličković et al., ICLR 2018), DistilBERT (Sanh et al., 2019), SupCon (Khosla et al., NeurIPS 2020), focal loss (Lin et al., ICCV 2017), XGBoost (Chen & Guestrin, KDD 2016).

---

## Review / Non-Review Detection - Dataset Sources (for citation)

The **review-detector** corpus (`data/review_detection.csv`, 599,154 rows, balanced
50/50 review vs non-review) was assembled locally by `download_negatives.py` +
`build_review_dataset.py`. Every row's `source` column records its origin. All
texts were HTML/URL-stripped, deduped (exact, case-insensitive, including across
classes), and split 80/10/10 stratified by label.

> NOTE: licenses and citations below are best-effort. Verify the exact citation
> format and the current license on each linked page before publishing.

### Positive class - REVIEWS (label 1; 6 domains, ~50k each)

| Domain | Dataset / where we got it | Link | License | Citation (verify) |
|---|---|---|---|---|
| Amazon products | Amazon Fine Food Reviews (`data/Reviews.csv`), via Kaggle (orig. Stanford SNAP) | https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews | research use | McAuley & Leskovec, "From amateurs to connoisseurs...", WWW 2013 |
| Movies | IMDB 50K Movie Reviews (`data/IMDB Dataset.csv`), via Kaggle (orig. Stanford) | https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews / http://ai.stanford.edu/~amaas/data/sentiment/ | research use | Maas et al., "Learning Word Vectors for Sentiment Analysis", ACL 2011 |
| Yelp businesses | YelpZip + YelpNYC + YelpChi -> `data/yelp_split.csv` (our balanced split) | http://shebuti.com/collective-opinion-spam-detection/ | research use (academic) | Rayana & Akoglu, "Collective Opinion Spam Detection...", KDD 2015 |
| Games | Steam user reviews, HF `SirSkandrani/steam_reviews_clean` | https://huggingface.co/datasets/SirSkandrani/steam_reviews_clean | see HF card | Steam (Valve) user reviews, via dataset author |
| Books | Amazon Reviews 2023 - Books reviews, HF `McAuley-Lab/Amazon-Reviews-2023` (`raw/review_categories/Books.jsonl`) | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 | research use (cite paper) | Hou et al., "Bridging Language and Items for Retrieval and Recommendation", 2024 (arXiv:2403.03952) |
| Apps | Mobile app reviews, HF `sealuzh/app_reviews` | https://huggingface.co/datasets/sealuzh/app_reviews | see HF card | Grano et al., "Android Apps and User Feedback: A Dataset for Software Evolution and Quality Improvement", WAMA 2017 |

### Negative class - NON-REVIEWS (label 0; 10 sources, ~337k pooled then downsampled to balance)

Sources marked "matched" are same-topic descriptive counterparts to a review domain
(used as hard negatives).

| `source` | What it is | Dataset | Link | License | Citation (verify) |
|---|---|---|---|---|---|
| amazon_meta | product descriptions (5 categories) - matched to amazon reviews | Amazon Reviews 2023 metadata (`raw/meta_categories/meta_*.jsonl`) | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 | research use | Hou et al., 2024 (arXiv:2403.03952) |
| book_desc | book descriptions / blurbs - matched to book reviews | Amazon Reviews 2023 metadata (`meta_Books.jsonl`) | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 | research use | Hou et al., 2024 |
| app_software_desc | app / software descriptions - matched to app reviews | Amazon Reviews 2023 metadata (`meta_Software.jsonl`) | https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 | research use | Hou et al., 2024 |
| business_desc | local-business descriptions - matched to Yelp reviews | McAuley Google Local place metadata (`meta-California.json.gz`) | https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/ | research use | Li et al., "Personalized Showcases...", SIGIR 2023 (and Yan et al.) - verify per dataset page |
| cmu_movies | film plot summaries - matched to movie reviews | CMU Movie Summary Corpus | http://www.cs.cmu.edu/~ark/personas/ | CC BY-SA | Bamman, O'Connor & Smith, "Learning Latent Personas of Film Characters", ACL 2013 |
| steam_desc | game store descriptions - matched to game reviews | HF `FronkonGames/steam-games-dataset` | https://huggingface.co/datasets/FronkonGames/steam-games-dataset | CC BY 4.0 | FronkonGames Steam Games Dataset (Steam store data) |
| amazon_qa | product questions | HF `embedding-data/Amazon-QA` | https://huggingface.co/datasets/embedding-data/Amazon-QA | MIT (HF) | McAuley & Yang, "Addressing Complex and Subjective Product-Related Queries with Customer Reviews", WWW 2016 |
| squad | general-knowledge questions | HF `rajpurkar/squad` | https://huggingface.co/datasets/rajpurkar/squad | CC BY-SA 4.0 | Rajpurkar et al., "SQuAD: 100,000+ Questions...", EMNLP 2016 |
| ag_news | news articles / headlines | HF `fancyzhx/ag_news` | https://huggingface.co/datasets/fancyzhx/ag_news | academic / non-commercial | Zhang, Zhao & LeCun, "Character-level CNNs for Text Classification", NeurIPS 2015 (orig. AG corpus by ComeToMyHead / A. Gulli) |
| wikitext | encyclopedic prose | HF `Salesforce/wikitext` (config `wikitext-103-raw-v1`) | https://huggingface.co/datasets/Salesforce/wikitext | CC BY-SA 3.0 | Merity et al., "Pointer Sentinel Mixture Models", 2016 |

### Negative class - broad GENRE coverage (label 0; 8 genres, ~5k each)

Added so the gate rejects *any* non-review genre a user might paste, not just the
genres of our review sources. (The recipe miss in the generalisation test exposed
that the negatives were topic-complete but not genre-complete.) ~5k each; poetry
(~0.5k) is the only small one.

| `source` | Genre | Dataset | Link | License | Citation (verify) |
|---|---|---|---|---|---|
| how_to | recipes / instructions | `corbt/all-recipes` | https://huggingface.co/datasets/corbt/all-recipes | see HF card | community recipe compilation |
| dialogue | casual conversation | `Estwld/empathetic_dialogues_llm` | https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm | CC BY-NC 4.0 | Rashkin et al., "Towards Empathetic Open-domain Conversation Models" (EmpatheticDialogues), ACL 2019 |
| email | email correspondence | `aeslc` | https://huggingface.co/datasets/aeslc | Enron-derived | Zhang & Tetreault, "This Email Could Save Your Life..." (AESLC), ACL 2019 |
| legal | legal / regulatory | `coastalcph/lex_glue` (config `eurlex`) | https://huggingface.co/datasets/coastalcph/lex_glue | CC BY 4.0 | Chalkidis et al., "LexGLUE: A Benchmark Dataset for Legal Language Understanding", ACL 2022 |
| code | source code | `code_search_net` (config `python`) | https://huggingface.co/datasets/code_search_net | MIT | Husain et al., "CodeSearchNet Challenge", 2019 |
| fiction | fiction / narrative | `euclaise/writingprompts` | https://huggingface.co/datasets/euclaise/writingprompts | see HF card | Fan et al., "Hierarchical Neural Story Generation" (WritingPrompts), ACL 2018 |
| academic | scientific abstracts | `CShorten/ML-ArXiv-Papers` | https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers | arXiv terms | arXiv ML paper abstracts (compilation) |
| poetry | poetry / verse | `merve/poetry` | https://huggingface.co/datasets/merve/poetry | see HF card | poetry compilation |

### Assembled (our own) intermediate files
- `data/non_reviews.csv` - the pooled negative class (~337k rows) produced by `download_negatives.py`.
- `data/review_detection.csv` - the final balanced, split corpus (599,154 rows) produced by `build_review_dataset.py`.
- `data/MovieSummaries.tar.gz` - local cache of the CMU corpus download (used by the movie-plots collector).

### Notes for the data statement
- All sources are publicly and freely downloadable; **no Kaggle authentication is
  required** by the build scripts (Kaggle entries above are origin references only -
  we used Amazon Fine Food / IMDB CSVs already on disk, and stream the rest from
  HuggingFace or academic mirrors).
- Fake / computer-generated reviews are counted as **reviews** here: the task
  classifies *form* (an evaluation) not *authenticity*.
- The two softest licenses are **AG News** (academic-only) and the **Yelp spam**
  datasets (academic research use) - flag both in the thesis data statement.

---

## Review / Non-Review Detection - Results

**Methodology:** each model's decision threshold is tuned on the **validation** split
(max macro-F1); metrics are reported on the held-out **test** split (59,916 rows,
balanced 50/50). Splits are disjoint and deduped; HTML/URL artifacts removed. The
task is genuinely easy/well-separated (distinct evaluative voice vs descriptive /
interrogative / factual text), so high scores are expected - unlike the 66-69%
behavioral-label Yelp ceiling.

> Status: FINAL on the **genre-expanded** dataset (599,154 rows, 24 sources incl.
> 8 broad non-review genres). **All 10 models (8 sklearn + LSTM + DistilBERT)
> retrained on it.** `classification_report` rounds to 2 dp, so "1.00" means
> ">= 99.5%" - cite the exact macro-F1.

### Deep-learning models (test split, genre-expanded data)

| Model | Config | Val macroF1 | Test macroF1 | Thr | Notes |
|---|---|---|---|---|---|
| LSTM (BiLSTM 128 + GloVe 300d + GlobalMaxPool) | MAX_LEN 250, batch 256, lr 5e-4 | 98.2% | **98.0%** | 0.57 | best epoch 5/15 (early stop @8, restored e5); **adding 8 genres cost ~0 accuracy** (also 98.0% pre-genre); per-class P/R ~0.98 |
| DistilBERT (LLRD + cosine + early stop) | MAX_LEN 256, batch 64 | 99.5% | **~99.5%** | 0.19 | **genre-trained**; plateaued by epoch 4, ran to the cap of 7 (each epoch a hair better); adding the 8 genres cost only ~0.1% vs the 99.6% pre-genre; train loss ~3e-4 = converged |

### sklearn models (test split, genre-expanded data, threshold-tuned on val)

| Model | Thr | Test macroF1 |
|---|---|---|
| MLP (1 hidden layer, 100) | 0.50 | **97.67%** |
| Logistic Regression | 0.46 | 96.37% |
| XGBoost (200 trees) | 0.41 | 95.13% |
| RandomForest (100 trees, depth-bounded*) | 0.50 | 92.39% |
| Naive Bayes | 0.51 | 93.39% |
| KNN (k=5 cosine, 20k subsample) | 0.41 | 92.39% |
| Decision Tree | 0.50 | 88.39% |
| AdaBoost (50 stumps) | 0.49 | 82.36% |

Ensemble ladder (test macroF1): AdaBoost 82.4 < DT 88.4 < KNN 92.4 ≈ RF 92.4* < NB 93.4 <
XGB 95.1 < LR 96.4 < MLP 97.7 < LSTM 98.0 < DistilBERT ~99.6. The
memory-efficient trainer fits the TF-IDF once per feature-size (float32) and is
behavior-preserving vs the per-model version.

*RF retrained depth-bounded (`max_depth=40, min_samples_leaf=5`, `shrink_rfs.py`) → **797 MB → 114 MB (7×)**; test macro-F1 94.82→92.39 (−2.4pp). Immaterial to the majority-vote gate, which is carried by DistilBERT (~99.6%), LSTM (98.0%), MLP (97.7%), LR (96.4%).

### Generalization sanity check (out-of-source, hand-written)
To confirm the high test scores reflect real review-ness and not memorised source
fingerprints, models were run on hand-written examples from **no** training source,
including adversarial **matched pairs** (a book *review* that summarises the plot vs a
bare book *plot summary*; enthusiastic *marketing* copy vs genuine reviews).

| Model | Score (out-of-source) |
|---|---|
| KNN (weakest, 10 examples) | **8/10** - 2 misses were subtle narrative reviews at P=0.40, just under threshold |
| LSTM (20 examples incl. hard pairs) | **18/20** |
| DistilBERT (20 examples incl. hard pairs) | **18/20** |

**Key finding:** both DL models correctly separated the *book-review-with-plot*
(review) from the *bare plot-summary* (non-review) - i.e. they key on **evaluation**,
not topic. This is direct evidence the ~98-99.6% test scores are genuine, not source
artifacts.

**Instructive misses (and one nameable limitation):**
- **Recipe / instructional text** ("Mix two cups of flour with one teaspoon of baking
  soda...") was misclassified as a review by **both** DL models. This is
  **out-of-distribution**: the negative class has descriptions, questions, news,
  plots, and encyclopedic text but **no how-to / instructional** text, and food
  vocabulary overlaps food reviews. -> Concrete limitation; closeable by adding an
  instructions/recipe negative source if needed.
- **LSTM** was fooled by enthusiastic **marketing** copy ("award-winning blender...
  powerful 1200-watt motor", P=0.66) while **DistilBERT rejected it** correctly
  (P=0.00) - the transformer separates promotional description from genuine
  evaluation better.
- **DistilBERT** missed one subtle review ("battery life blew me away...", P=0.11)
  that LSTM caught (P=0.92) - a borderline phrasing that reads spec-like.

### Genre gap closed (after adding the 8 broad genres)
After retraining on the genre-expanded dataset, the same out-of-source probe across
all 8 genres (recipe, code, legal, email, fiction, academic, dialogue, poetry):
- **LR: 10/10**, **LSTM: 10/10**, **DistilBERT (genre-trained): 10/10** (all genres
  P=0.00, reviews P=1.00 - recipe/fiction/dialogue all confidently rejected). The
  old pre-genre DistilBERT was the lone model that called the recipe a review;
  retraining fixed it, so every model in the gate now handles all 8 genres.
- **5k examples per genre was sufficient** - the earlier fiction/dialogue failures
  were a KNN weakness (nearest-neighbour on word overlap), not a data shortage;
  every stronger model learns them. KNN remains weak on fiction/dialogue but is the
  weakest of ~10 voters and is outvoted.

### Dashboard "Analyze" feature - end-to-end verified
`POST /analyze` (gate -> fake + sentiment) tested in-process:
- single review text -> REVIEW 10/10 -> GENUINE, POSITIVE.
- CSV (4 rows, 75% reviews) -> status ok + warning; a **recipe row was correctly
  excluded** (3/10, not a review), the 3 reviews scored (fake + sentiment); under
  50% would have errored with nothing analyzed.

---

## Broadened Fake-Review (CG/OR) Detection - Dataset Sources (for citation)

`data/fake_reviews_broadened.csv` (110,698 rows after adding `openai_gpt`; balanced 50/50) extends the original
Salminen CG/OR data with **modern, multi-generator** machine-written reviews and a
**movie domain**. Built by `build_broadened_cgor.py`; the original
`data/fake reviews dataset.csv` is left untouched.

**Label:** 1 = CG (machine-generated / fake), 0 = OR (original / human). Domain-matched
(products + movies in identical proportions in both classes -> no domain shortcut),
deduped, HTML/URL stripped, and **length-balanced**: machine reviews (sutro/Lyra)
skew much longer than human ones, so we keep ALL the CG and draw an abundant human
pool (full Amazon + all IMDB) matched to the CG per domain+length-bin (94% of CG
retained) — this dropped *length-alone* accuracy from **70% to 52%** (so the model must
learn machine-vs-human from content, not length). Stratified 80/10/10 split.

> Verify each license/citation on its page before publishing.

| `source` | Class · Domain | Dataset / where | Link | License | Citation (verify) |
|---|---|---|---|---|---|
| salminen_cg / salminen_or | CG & OR · products | original Salminen fake-reviews set (`data/fake reviews dataset.csv`) | https://osf.io/tyue9/ (also Kaggle "Fake Reviews Dataset") | research use | Salminen, Kandpal, Kamel, Jung & Jansen, "Creating and detecting fake reviews of online products", *J. Retailing and Consumer Services* 64 (2022) |
| sutro | CG · products | HF `sutro/synthetic-product-reviews-20k` | https://huggingface.co/datasets/sutro/synthetic-product-reviews-20k | see HF card | community LLM-generated product reviews |
| kenshiii | CG · products | HF `Kenshiii/synthetic-product-reviews` | https://huggingface.co/datasets/Kenshiii/synthetic-product-reviews | see HF card | community LLM-generated product reviews |
| lyra / lyra_human | CG & OR · movies | HF `Lyra-stellAI/AI_Human_generated_movie_reviews` | https://huggingface.co/datasets/Lyra-stellAI/AI_Human_generated_movie_reviews | see HF card | human + AI movie reviews; AI generators: GPT-4-turbo, GPT-3.5-turbo, Gemini Pro 1.5, fine-tuned Llama-3-8B |
| amazon_real | OR · products | Amazon Fine Food Reviews (`data/Reviews.csv`) | https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews | research use | McAuley & Leskovec, "From amateurs to connoisseurs...", WWW 2013 |

**Composition (after length-stratified balancing — 86,816 rows, 94% of CG kept):**
products 88.2% (CG: Salminen 20,040 + sutro 17,568 + Kenshiii 668;
OR: Amazon 35,011 + Salminen 3,265) · movies 11.8% (CG: Lyra AI 5,132; OR: IMDB 4,561 + Lyra
human 571). Keeping all CG and matching it against an abundant human pool (full Amazon
+ all IMDB) recovers the size to ~87k (94% of CG retained) while staying length-
balanced. Generator diversity (Salminen's older generation + sutro/Kenshiii +
Lyra's 4 LLMs) is deliberate, so the detector learns "machine-generated" rather than
one model's fingerprint.

**Honest limitation:** movies are only ~12% (Lyra is the sole AI-movie source via
"Route A" of public datasets); products dominate. It is domain-balanced *across
classes* so it is sound, but true multi-domain CG parity (restaurants / apps / books
/ games) would require generating AI reviews ourselves ("Route B").

### Broadened CG/OR - Results (test split, threshold-tuned on val)

#### Multi-domain (6 domains) — CURRENT / FINAL (supersedes the products+movies block below)

Built by the multi-domain `build_broadened_cgor.py`: **113,314 rows, 6 domains** (products/movies/restaurants/apps/books/games), 50/50, **10 generator families**. AI from open-model domains + GPT domains + W4 short (length-matched) + restaurant scale-up; human reviews per domain from `review_detection.csv`; per-domain CG cap (20k) + per-(domain,length-bin) matching. Audited: **per-domain length-AUC ~0.5, 0 leakage, no NaN/dupes/HTML**.

**Final dataset = 128,956 rows** after adding an **old/weak-LLM family** (distilgpt2/gpt2/gpt2-medium/OPT-350m, `generate_ai_oldgen_reviews.ipynb`) **across all 6 domains** — to give the GPT-2-era paradigm cross-domain siblings (it previously lived only in products). 14 generator families total.

**sklearn (test, threshold-tuned on val):** MLP **94.7%** · LR 94.2% · XGB 92.1% · RF 91.9% · NB 89.6% · KNN 87.4% · DT 82.9% · AdaBoost 74.0% (macro-F1). **LSTM 98.3%** (BiLSTM+GloVe, threshold 0.73, best epoch 14/30, early-stopped — mild controlled overfit, val plateau ~98.3%). **DistilBERT 99.4%** (LLRD+cosine+AMP recipe, threshold 0.87, best epoch 6/7 — **best Model-A member**). The DL models clear the sklearn ensemble (~95%), as expected for this separable text task. Model-A full ensemble: 8 sklearn (74–94.7%) + LSTM 98.3% + DistilBERT 99.4%, all ONNX/joblib. *(These are ~2pp below the pre-old-gen numbers — expected: weak/old-LLM text is less polished and more human-like, so genuinely harder to classify. The dataset is now harder + more robust, covering the weak end of the generator spectrum.)*

**Generalization (LR probe; the "catch-all" metrics), with old-gen added:**
- **Leave-one-generator-out: mean 0.79** across 14 families. Modern families strong (gpt 1.00, llama3 0.97, zephyr 0.94, yi 0.93, qwen 0.91, openai_gpt 0.89, sutro 0.87, lyra 0.83); old/weak families moderate (distilgpt2 0.76, gpt2 0.73, opt350 0.66, gpt2med 0.64); laggard **salminen_cg 0.15** (up from 0.07 — old-gen siblings helped, but Salminen's GPT-2-fine-tuned-on-reviews style remains the hardest unseen case).
- **Leave-one-domain-out: mean 0.77.** movies 0.97, books 0.93, games 0.87, apps 0.83, restaurants 0.70; **products 0.33** (still dragged by salminen/sutro/kenshiii — products-only *datasets* that can't be relocated).
- **Honest takeaway:** excellent generalization across **modern LLM** families/domains (0.83–1.00, which is what real fake reviews use); reasonable across old/weak LLMs (0.64–0.76); the persistent hard floor is the legacy products-only synthetic corpora (Salminen GPT-2, sutro, Kenshiii). These are **in-distribution** in deployment (caught fine); the low leave-one-out numbers only describe a never-seen *legacy* generator.

**HC3 cross-dataset eval (`evaluate_cgor_external.py`, zero-shot on Human-vs-ChatGPT — a different corpus AND text type: Q&A answers, not reviews; 8,000 balanced texts):**

| model | HC3 macro-F1 | ChatGPT recall | human spec |
|---|---|---|---|
| **DistilBERT** | **90.1%** | 87.8% | 92.4% |
| sklearn (knn/rf/dt/lr/nb/xgb/adaboost/mlp) | 42–48% | 12–26% | 77–94% |
| LSTM | 40.1% | 7.2% | 97.4% |
| naïve 10-model majority | 45.0% | 13.2% | 95.3% |

**Key finding:** **DistilBERT transfers (90% zero-shot on a domain+format it never saw); the TF-IDF sklearn models and GloVe-LSTM do not (~40–48%).** The bag-of-words / static-embedding models learned *review-specific surface features* (product/review vocabulary) that don't exist in Q&A text, so they collapse to "predict human". DistilBERT learned a deeper, **transferable machine-generated-text representation** — it recognises AI writing across domain and format. → For the Dashboard, **trust DistilBERT (and the contextual model) on out-of-distribution input**; a naïve equal-weight majority is dragged down by the surface-feature models. (HC3 is the *hardest* transfer — Q&A, not reviews; real uploads are reviews gated by the review-detector, where all models are in-distribution.)

---
#### Earlier (products + movies only, 86,816 rows) — for reference

Trained on `fake_reviews_broadened.csv` (86,816 rows, length-balanced). Test split
8,682 rows (4,341 per class). 1 = CG (machine), 0 = OR (human).

| Model | Val macroF1 | Test macroF1 | Thr | Notes |
|---|---|---|---|---|
| LSTM (BiLSTM 128 + GloVe 300d) | 98.7% | **99.0%** | 0.74 | best epoch 10/15 (early stop @13, restored e10); train 99.6% vs val 98.6% = mild controlled overfit; per-class P/R 0.98-0.99 |
| DistilBERT | - | _pending_ | - | `train_fake_broadened_distilbert.ipynb` |

sklearn (test split, threshold-tuned on val):

| Model | Thr | Test macroF1 |
|---|---|---|
| MLP (1 hidden, 100) | 0.52 | **97.16%** |
| Logistic Regression | 0.46 | 96.54% |
| XGBoost (200 trees) | 0.47 | 96.07% |
| RandomForest (100 trees) | 0.54 | 95.46% |
| Naive Bayes | 0.32 | 94.27% |
| KNN (k=5 cosine, 20k subsample) | 0.61 | 92.43% |
| Decision Tree | 0.05 | 90.22% |
| AdaBoost (50 stumps) | 0.51 | 82.93% |

Ladder (test macroF1): AdaBoost 82.9 < DT 90.2 < KNN 92.4 < NB 94.3 < RF 95.5 <
XGB 96.1 < LR 96.5 < MLP 97.2 < LSTM 99.0 (< DistilBERT, pending).

**Caveat (AI-text detection) - VERIFIED generalization failure:** the high in-dist
scores largely reflect **per-generator / source-topic memorization**, NOT general
machine-detection. Leave-one-generator-out test (LR): holding `sutro` out as an
*unseen* generator, its CG-recall collapses **99.8% -> 2.2%** and overall test acc
drops **96.5% -> 77.7%** (Salminen/Lyra, still in training, stay ~93-99%). So the
~97-99% **overstates** real-world detection of AI reviews from *new* LLMs.
Contributing factors: (1) CG vs OR sources differ in topic (e.g. sutro general-product
CG vs `amazon_real` FOOD OR), so part of the signal is topic/vocabulary, not
machine-ness; (2) AI-text detectors intrinsically transfer poorly across generators.
It is NOT a training bug — dedup (0 dups) so no split leakage, length neutral (52%
length-alone), honest threshold. Conclusion: treat the broadened detector as
**in-distribution only**; the headline F1 is genuine on these generators but not a
robust general "is-this-AI-written" capability. (A useful thesis finding in itself.)

**Leave-one-generator-out baseline** (LR @0.5, via `evaluate_cgor_generalization.py`):

| held-out generator | in-dist recall | LOGO recall (unseen) | specificity |
|---|---|---|---|
| kenshiii (~81) | 1.00 | 0.79 | 0.96 |
| lyra | 1.00 | 0.02 | 0.96 |
| salminen_cg (GPT-2) | 0.94 | 0.01 | 1.00 |
| sutro | 1.00 | 0.02 | 0.96 |

Mean LOGO recall ~**0.21** — i.e. an unseen generator is caught only ~21% of the time,
vs 96.6% in-distribution. Specificity stays 0.96-1.00 (misses are false negatives, not
over-flagging). salminen_cg (GPT-2) evades a model trained on GPT-4-class text, so
**generator-family diversity is the key lever**. This ~0.21 is the yardstick to beat
via Route B (more generators + topic-matched generation); re-run the script after each
addition to track it.

**After Route B step 1** (added `openai_gpt`: 12k topic-matched, multi-model/
temperature/persona/style GPT reviews via `generate_ai_reviews.py`):

| held-out generator | LOGO recall (baseline) | LOGO recall (after Route B) |
|---|---|---|
| sutro | 0.02 | **0.66** |
| lyra | 0.02 | 0.14 |
| kenshiii | 0.79 | 0.79 |
| salminen_cg (GPT-2) | 0.01 | 0.06 |
| openai_gpt (new, held out) | - | 0.54 |

Mean LOGO recall **0.21 -> 0.44** (more than doubled), specificity ~0.96, dataset
86.8k -> **110,698** rows. Biggest jump: `sutro` 0.02 -> 0.66 — the diverse,
topic-matched GPT data lets the model recognise *other GPT-class* product generators.
`salminen_cg` (GPT-2) stays ~0.06: an old, different family still evades, so the next
lever is **non-GPT families** (Llama / Mistral / Qwen via Colab). Lesson confirmed:
generator-family diversity drives cross-generator generalization, and topic-matched
generation is an effective, cheap way to add it.

**After Route B step 2** (added 2 non-GPT families: Qwen 2,999 + Llama-3 3,000, open
models via `generate_ai_reviews_open.ipynb`; dataset now 122,696 rows):

| held-out generator | LOGO recall (unseen) |
|---|---|
| llama3 | 0.96 |
| openai_gpt | 0.91 |
| qwen | 0.90 |
| kenshiii | 0.80 |
| sutro | 0.74 |
| lyra (movies) | 0.15 |
| salminen_cg (GPT-2) | 0.07 |

Mean LOGO recall **0.44 -> 0.648** (baseline 0.21). Modern LLM families now
cross-generalise strongly (held-out modern generators 0.74-0.96) - the model learned
general machine-ness. Two laggards remain, and they are *diagnostic*: (1) **lyra 0.15**
- it is the ONLY movie CG source, so holding it out leaves zero machine-movie data =
a DOMAIN-coverage gap (fixed by adding more movie / other-domain CG), not a generator
issue; (2) **salminen_cg 0.07** = GPT-2, an old family modern LLMs do not resemble
(edge case; GPT-2 is not a realistic current fake-review threat). Trajectory of the
intervention: **0.21 -> 0.44 -> 0.65** as generator families were added.

---

### My Uploads — per-user persistence of Analyze results (stateful personalization)

**Motivation.** The app shipped with a sign-up/login feature (`auth.py`) that was effectively
decorative: every section served the same shared Amazon/Yelp data, so being logged in changed
nothing. **My Uploads** makes the account meaningful — each user now has a private, persistent
history of everything they have analysed in the Dashboard → Analyze box.

**What it does.** Whenever a logged-in user analyses input in the Analyze box (pasted text, a
CSV, and/or image screenshots), every resulting per-review item — reviews **and** non-reviews —
is saved to the database under their username, with a short **review id** and an **upload
timestamp**. A new **"My Uploads"** source appears in both the **Fake Review Detection** and the
**Sentiment Analysis** sections (it is the **default** landing tab in each), where the user can
browse, search, filter, and delete their saved results. The same stored record renders its
**fake** half in the Fake section and its **sentiment** half in the Sentiment section.

**Persistence model (SQLite, `data/users.db`).** Two tables added to the existing auth DB:
- `analyses` — one row per analysed item: `username, review_id, kind ('text'|'image'|'csv'),
  source, csv_batch, text, is_review, result_json (the full /analyze item dict), uploaded_at`.
- `csv_uploads` — one row per uploaded CSV file: `batch_id, username, filename, meta_json
  (status/warning/rows/review_rows/review_pct), uploaded_at`, so a CSV's rows stay grouped.

Records are keyed by **username**, not by session token, so they **survive server restarts**
(the in-memory sessions reset on restart; the DB persists). The full `/analyze` item dict is
stored verbatim, so **no information is lost** — per-model fake/sentiment votes, the
review/non-review verdict, phrase/sentence highlights, within-batch duplicate flags
(exact/near), and the CSV-level coordinated-reviewer (`csv_behavior`) signals all reload intact.

**Backend (`auth.py` + `main.py`).** Helpers `save_upload`, `get_uploads`, `delete_item`,
`delete_csv_group` (all scoped to the user). Endpoints: `GET /personal-results` (returns
`{text, images, csv_groups}`), `POST /personal-results/delete-item` (by `review_id`),
`POST /personal-results/delete-csv` (by `batch_id`, removes all of that CSV's rows + the group
record). `/analyze` resolves the user from the `session_token` cookie and calls `save_upload`;
persistence failures are swallowed so they can never break analysis. Guards: missing session →
401, missing id → 400.

**Frontend (`static/index.html`).** The My Uploads view mirrors the Analyze layout — a **Text**
heading, an **Images** heading, then **one group per uploaded CSV file** (`CSV · filename ·
🕑 timestamp` + the rows/reviews/non-reviews/% strip). Text and image cards each show their own
upload timestamp; CSV reviews are grouped under their file's heading and timestamp, so per-CSV
duplicate/coordination context is preserved. Controls: a **live search** box (filters by review
id or text as you type), an **order** selector (latest/earliest), **type filters** (show/hide
Text, Images, CSV), and a mode-dependent **stats bar** (Fake: Total · Flagged FAKE · Genuine ·
Non-reviews; Sentiment: Total · Positive · Negative · Non-reviews). **Delete:** a button at the
bottom of each text/image card and at the end of each CSV group heading; deletion is
section-independent (it removes the record from the database, so it disappears from both views).
The card renderer (`makeAnalyzeCard`) is parameterised with a `mode` ('fake' | 'sentiment') so
one record yields a fake-only or sentiment-only card.

**Why it matters for the thesis.** It is the one place the otherwise **stateless** forensics
service (designed for horizontal scaling — see §4.3 non-functional requirements) becomes
**stateful and multi-user**: a deliberate, scoped exception (a per-user audit trail) layered on
top of the stateless scoring core, demonstrating that the detector can back a personalised,
revisitable review-history product without changing the scoring pipeline itself.

---

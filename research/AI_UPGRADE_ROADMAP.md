# KalaSetu AI Upgrade Roadmap

Branch: `upgrade/kalasetu-ai-suite`

This document separates code-level improvements from algorithm changes that need real data and empirical validation. No accuracy improvement is claimed by this branch.

## Changes committed in this branch

| Module | Existing approach | Change in this branch | Status |
|---|---|---|---|
| Fair pricing | XGBoost regressor; preprocessing/feature handling was not consistently isolated from splitting | Candidate comparison scaffolding for XGBoost, Random Forest and CatBoost; preprocessing inside the model pipeline; train/validation/test partition; synthetic-data warning | Code committed; local training and API integration tests pending |
| Demand prediction | XGBoost regressor; one-hot encoder was fitted before the train/validation/test split; no reported held-out test metrics | Encoder now fits on training rows only; validation and reserved test metrics are reported separately; metadata documents random split limitation and synthetic-data status | Code committed; local training and inference tests pending |
| Anomaly detection | Isolation Forest trained on all rows, including rows labelled anomalous; training-only scores were reported | Stratified holdout; fit Isolation Forest only on labelled-normal training observations; evaluate labelled holdout with precision, recall, F1, average precision and confusion matrix | Code committed; local training/inference tests pending |

## Full module inventory: old approach and recommended direction

| # | Module | Existing implementation | Recommended direction | Data needed before claiming model improvement |
|---:|---|---|---|---|
| 1 | Fair Pricing | XGBoost regression | Benchmark XGBoost, CatBoost and Random Forest with identical splits; choose by validation MAE/RMSE, then report untouched test results | Real artisan product quotes and accepted sale prices; material, labour, overhead, location, craft, dimensions and date |
| 2 | Demand Prediction | XGBoost regression | Compare XGBoost/CatBoost against seasonal-naive and moving-average baselines; chronological backtesting | Dated product-level sales/orders, stock availability, price, promotions, season and region |
| 3 | Buyer Recommendation | Weighted rule-based matching | Explainable content-based ranking first; collaborative filtering only after enough buyer-product interaction history | Impressions, clicks, enquiries, saves, purchases, buyer preferences and timestamps |
| 4 | Product Cataloguing | EfficientNet-B0, three classification heads (category/material/color) | Benchmark EfficientNet-B0 against MobileNetV3; class-balanced fine-tuning and augmentation; retain pretrained baseline | Current CSV has only 12 images. Build a diverse, correctly labelled image set with artisan/product-level split |
| 5 | Anomaly Detection | Isolation Forest | Normal-only fit plus labelled evaluation; later compare LOF and supervised models only if reliable labels exist | Real transaction/order/account events, confirmed fraud/abuse outcomes and time-aware splits |
| 6 | Inventory Intelligence | XGBoost classifier | First remove circular label construction; use interpretable stock policy as baseline; train a classifier only on observed stockout/reorder outcomes | Historical inventory snapshots, sales, stockouts, lead times, replenishments and lost sales |
| 7 | Smart Logistics | K-Means clustering + OR-Tools VRP | Keep OR-Tools as optimizer; benchmark geographic clustering/distance calculations and add capacity/time-window constraints | Real order coordinates, vehicle capacities, delivery windows, road-distance/time data |
| 8 | Market Intelligence | Weighted scoring rules | Keep explainable scoring until real market outcomes exist; then compare supervised ranking/regression | Real market observations, demand/sales outcomes, source/date/geography and product categories |
| 9 | NLP / Semantic Intelligence | TF-IDF text vectors | Benchmark multilingual sentence embeddings against TF-IDF; support Indian languages and semantic retrieval | Human relevance judgements for artisan product/query pairs in supported languages |
| 10 | OCR | Tesseract + OpenCV preprocessing | Benchmark Tesseract against PaddleOCR/EasyOCR on the same artisan-image set; select by CER/WER, language support and latency | Images of labels, invoices and product descriptions with verified transcriptions |
| 11 | Verification | Pretrained EfficientNet-B0 embeddings + cosine similarity | Benchmark CLIP/DINOv2 or fine-tuned metric learning; calibrate threshold on held-out pairs | Same-item and different-item image pairs, including lighting/viewpoint variation |
| 12 | Voice Intelligence | SpeechRecognition/Google service + language detection + rule-based intent | Benchmark Whisper and Indic ASR options; select by WER/CER, latency, privacy and offline needs | Recorded consented speech with verified transcripts for target Indian languages |
| 13 | Waste Intelligence | Weighted rule-based reuse scoring | Keep explainable scoring; learn ranking only after reuse outcomes are recorded | Waste material, condition, processing cost, reuse outcome and resulting product sales |
| 14 | Sustainability | Weighted score and fixed carbon factors | Source/version emission factors, geography and uncertainty; keep score components explainable | Verified material, energy, packaging and transport activity data plus factor references |
| 15 | Digital Product Passport | Structured passport + SHA-256 integrity check | Improve schema validation, issuer/provenance evidence and tamper checks; this is not an ML task | Verified artisan, material-origin, process and custody records |

## Evaluation rules

- Pricing: MAE, RMSE, MAPE where target values are non-zero; compare against transparent cost baselines.
- Demand: MAE/RMSE and seasonal-naive baseline under chronological backtesting.
- Cataloguing: macro-F1, per-class precision/recall, confusion matrices and a product-level held-out test set.
- Anomaly: precision, recall, F1, average precision/PR-AUC and false-positive review burden.
- Recommendation/NLP: Recall@K, Precision@K, NDCG@K and MAP@K from human or behavioural relevance labels.
- OCR/ASR: character error rate, word error rate and latency.
- Verification: ROC-AUC, PR-AUC, FAR, FRR and EER; choose thresholds on validation data only.
- Logistics: distance/cost, constraint violations and on-time delivery against a defined baseline.
- Sustainability: traceable emission factors and uncertainty; do not report a deterministic score as a trained ML prediction.

## Important limitations

1. The bundled pricing, demand, anomaly and cataloguing datasets are synthetic or extremely small. They are suitable for pipeline checks, not claims of real-world accuracy.
2. The cataloguing CSV contains 12 labelled images. A deeper architecture alone will not fix insufficient training data.
3. Several modules are decision-support engines rather than trained ML models. Do not add an ML algorithm just to make every module look like machine learning.
4. A commit is not proof of execution. The changes on this branch still require local dependency installation, model training, inference smoke tests and backend endpoint tests.
5. Do not merge this branch into `main` until those tests pass and generated model artifacts are reviewed.

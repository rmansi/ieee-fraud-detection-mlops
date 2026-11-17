# IEEE-CIS Fraud Detection — End-to-End MLOps Pipeline
***

I built this project to learn what it takes to move a fraud model from a notebook to something closer to production. It covers the full path: exploring the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) data, training and tracking models with MLflow, serving predictions through a FastAPI service in Docker, and monitoring that service with Prometheus and Grafana. Along the way I tried to check my assumptions with data instead of trusting them, and a few of the most useful lessons came from exactly those checks.

## Table of Contents

1. [Overview](#1-overview)
2. [Problem and Data](#2-problem-and-data)
3. [What I Learned from EDA](#3-what-i-learned-from-eda)
4. [Methodology](#4-methodology)
5. [Results](#5-results)
6. [Serving](#6-serving)
7. [Monitoring](#7-monitoring)
8. [Testing and CI](#8-testing-and-ci)
9. [How to Run](#9-how-to-run)
10. [Project Structure](#10-project-structure)
11. [Limitations and Next Steps](#11-limitations-and-next-steps)
12. [Credit](#12-credit)

---

## 1. Overview

| | |
|---|---|
| **Task** | Binary classification: is a card transaction fraudulent? |
| **Data** | IEEE-CIS Fraud Detection (Kaggle), 590,540 labeled transactions |
| **Best model** | XGBoost — AUC-PR 0.578, ROC-AUC 0.916 on a future time window |
| **Serving** | FastAPI `/predict` in Docker, ~0.13 s per request on my laptop |
| **Monitoring** | Prometheus metrics with a provisioned Grafana dashboard |
| **Quality** | 13 pytest tests and ruff linting, run by GitHub Actions on every push |

```
Kaggle CSVs ──► make_parquet ──► build_features ──► train (logreg / XGBoost)
                                 (merge, n_missing,      │  logs params, metrics,
                                  categories,            ▼  columns & categories
                                  time split)       MLflow tracking + model registry
                                                         │
                                                    export_model ──► model_store/
                                                                         │
                         docker compose ┌────────────────────────────────┘
                                        ▼
                     FastAPI  /predict /health /metrics ──► Prometheus ──► Grafana
```

## 2. Problem and Data

| Property | Value |
|---|---|
| Transactions (train) | 590,540 |
| Transaction columns | 394 |
| Identity columns | 41 (joined on `TransactionID`) |
| Transactions with identity info | 144,233 (24.4%) |
| Fraud rate | 3.50% |
| Time span | 182 days |

What makes this dataset tricky is the combination of a small positive class, heavy and *structured* missingness, mostly anonymized features (`V1`–`V339`, `C`, `D`, `M`), and a fraud rate that moves around over time.

Accuracy isn't a useful metric here: since 96.5% of transactions are legitimate, a model that always says "not fraud" would score 96.5% accuracy while catching no fraud at all.

## 3. What I Learned from EDA

1. **The positive class is small.** About 3.5% of transactions are fraud, so I evaluate with AUC-PR and recall at a fixed precision rather than accuracy.

2. **The fraud rate drifts over time.** The daily rate ranges from roughly 1% to 7%, and the first ~25 days are noticeably lower. Interestingly, the first 80% and last 20% of the data have almost the same overall rate (3.51% vs 3.44%), so the drift only shows up at the daily level. I still split by time, because that mirrors how the model would actually be used — trained on the past, scoring the future — and a random split would let information from future days leak into training.

3. **Missing values carry signal.** 168 of the 394 transaction columns are more than 70% empty, but the missingness isn't random: rows with the fewest blanks had 8.0% fraud, compared with roughly 1.6–2.9% for everything else. The relationship isn't monotonic either (the middle group was the safest), which suits tree models better than linear ones. I kept the NaNs as they are, since filling them with averages would erase this pattern.

4. **The "identity" effect is mostly product type.** At first, having identity info looked like a strong fraud signal (7.85% vs 2.09%). Looking closer, product W — 74% of all transactions, with about 2% fraud — *never* has identity info, while the other products almost always do. So these two patterns largely describe the same split, which is a good example of confounding.

5. **…but identity still matters a little on its own.** Within product C alone, rows with identity info had 12.3% fraud vs 5.8% without. The second group is small (~6.3k rows), and this is an association rather than a cause, but it was enough for me to keep both features.

| Finding | What I decided |
|---|---|
| Small positive class | AUC-PR as the primary metric |
| Drift over time | Time-based split, never random |
| Informative missingness | Keep NaNs; add an `n_missing` feature |
| Identity / product overlap | Keep both `ProductCD` and `has_identity`, without treating them as independent evidence |

The full analysis is in `02_eda.ipynb`.

## 4. Methodology

### 4.1 Data preparation
- I converted the CSVs to Parquet and downcast `float64 → float32` and integers to their smallest fitting type. In memory, the transaction table went from 1,879 MB to 978 MB; on disk, from a 683 MB CSV to a ~79 MB Parquet file, thanks to Parquet's columnar compression.
- The transaction and identity tables are left-joined, so no transaction is dropped. I used pandas' `validate="one_to_one"` so the merge fails loudly if an ID is ever duplicated.

### 4.2 Features

| Feature | Definition |
|---|---|
| `has_identity` | Whether the transaction appears in the identity table |
| `n_missing` | Number of blank feature values in the row (computed per row, so it can't leak information across the split) |
| 31 text columns | Converted to pandas `category` dtype; the training category lists are saved with the model |

### 4.3 Validation strategy
- Rows are sorted by `TransactionDT`, and the **most recent 20%** is held out: 472,432 rows for training (3.51% fraud) and 118,108 for validation (3.44% fraud). The split function refuses unsorted input, since a silently unsorted split would leak future data.
- I leave `TransactionID` and `TransactionDT` out of the model inputs. The ID has no real meaning, and every validation timestamp is later than anything in training, so patterns learned from raw time wouldn't carry forward.

### 4.4 Models

| Model | Inputs | Preprocessing |
|---|---|---|
| Logistic regression (baseline) | 402 numeric columns | Median imputation and standard scaling, fitted on training data only |
| XGBoost | All 433 columns | None — XGBoost learns a default branch direction for missing values at each split, and handles categoricals natively (`enable_categorical=True`) |

The XGBoost settings (500 trees, learning rate 0.05, max depth 8, row/column subsampling of 0.8/0.5, `tree_method="hist"`) are sensible starting values. I haven't tuned them yet.

### 4.5 Metrics

| Metric | What it tells me |
|---|---|
| **AUC-PR** (primary) | How well the model finds fraud without raising too many false alarms. A random model scores roughly the fraud rate (~0.034 here). |
| ROC-AUC | How well the model ranks fraud above legitimate transactions. Useful, but it can look optimistic on rare events, so I don't rely on it alone. |
| Recall @ 90% precision | If only flags that are right 9 times out of 10 are allowed, what share of all fraud gets caught? |

## 5. Results

All scores are on the **most recent 20%** of transactions, which the model never saw during training.

| Model | ROC-AUC | AUC-PR | Recall @ 90% precision |
|---|---|---|---|
| Random guessing | 0.500 | 0.035 | 0.0% |
| One-feature rule (`n_missing`) | 0.700 | 0.085 | 0.0% |
| Logistic regression (numeric features, median imputation) | 0.812 | 0.177 | 0.0% |
| **XGBoost** (all features, NaNs kept, native categoricals) | **0.916** | **0.578** | **26.0%** |

A few things I noticed:
- **XGBoost's AUC-PR is about 3.3× the logistic regression baseline** and roughly 17× random guessing.
- **At 90% precision, XGBoost catches about a quarter of all fraud.** Logistic regression never reaches 90% precision at any threshold, so it can't be used for a high-confidence alerting tier.
- **The results are reproducible.** Every run is tracked in MLflow, and retraining with the same settings and `random_state` gives identical scores.
- **The engineered features turned out to matter less than I expected.** By gain importance (the average loss reduction a feature contributes when used in a split), the top features are anonymized `V` columns, while `n_missing` ranks 292nd and `has_identity` 397th of 418. My understanding is that XGBoost can see *which* individual columns are blank, so a summary count becomes mostly redundant. It was a useful reminder that a feature that helps a simple model may add little to a more flexible one.

## 6. Serving

### 6.1 Model lifecycle
1. `train.py` logs parameters, metrics, the model, `columns.json`, and `categories.json` to MLflow, and registers the model as `fraud-xgboost`.
2. `export_model.py` copies the latest registered version and its metadata into a self-contained `model_store/` folder. I needed this step because MLflow records absolute artifact paths from the machine that trained the model, and those Windows paths don't exist inside a Linux container.
3. The API loads only from `model_store/` (configurable through the `MODEL_DIR` environment variable), so it behaves the same way on my laptop and in Docker.

### 6.2 Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Fraud probability for one transaction |
| `GET /health` | Liveness check and the loaded model version |
| `GET /metrics` | Prometheus metrics |

### 6.3 Keeping training and serving consistent
This was the part I spent the most time on, since mismatches here fail silently. For each request, the API:
- rebuilds all 433 training columns in training order, treating any feature that isn't sent as missing;
- maps category values onto the *training* category lists (prepared once at startup as `CategoricalDtype`s), turning unseen values into blanks explicitly;
- computes `n_missing` and `has_identity` itself, reusing the same `add_features` function as training — clients aren't allowed to send them;
- rejects unknown feature names (typos), engineered features, and text in numeric fields with a clear 422 error.

**How I verified it:** I sent 20 real validation transactions to the API and compared its answers with the model's direct predictions. The largest difference was about 4×10⁻⁹, which is just float rounding. I repeated this check after each change to the API and again inside Docker, and it's now an automated test.

**Category handling, tested rather than assumed:** I was worried that a single request would produce different category codes than training (for example, `visa` being code 3 in training but code 0 in a one-row DataFrame). In my test, XGBoost 3.2 matched categories by name, so the predictions agreed. I still pass the training category lists explicitly, for two reasons I found while testing: rebuilding categories from one row crashed XGBoost whenever a category column was blank, and pandas warns that unseen categories will raise an error in a future version.

### 6.4 Latency
After adding Prometheus metrics, I noticed that each prediction was taking about 0.86 s, which felt far too slow for a single row. Timing each stage showed that preparing the request took around 80% of that, because I was assigning the DataFrame column by column — 433 separate updates. Building all the values in a dictionary first and creating the DataFrame once made a big difference:

| Stage | Before | After |
|---|---|---|
| Request preparation | ~0.70 s | ~0.04 s |
| Total per request (laptop) | ~0.86 s | ~0.13 s |

I also set `n_jobs=1` for inference, since spinning up multiple threads for a single row likely costs more than it saves (the measured gain was small and within noise, but it also keeps concurrent requests from competing for every core). After both changes, I re-ran the consistency check to make sure the predictions hadn't changed.

## 7. Monitoring

The API exposes four custom metrics through `prometheus_client`, and Prometheus scrapes them every 5 seconds. Grafana's datasource and dashboard are provisioned from files in the repo, so they come back automatically after the containers are rebuilt.

| Panel | Query | What it shows |
|---|---|---|
| Predictions per second | `rate(fraud_predictions_total[1m])` | How busy the service is |
| Rejected requests per second | `rate(fraud_requests_rejected_total[1m])` | Whether clients are sending invalid data |
| p95 latency | `histogram_quantile(0.95, rate(fraud_prediction_latency_seconds_bucket[5m]))` | How long the slowest 5% of requests take (estimated from histogram buckets) |
| Average fraud probability | `rate(fraud_probability_sum[5m]) / rate(fraud_probability_count[5m])` | Whether the model's output is shifting |

When I replayed validation transactions through the API, p95 latency settled around 0.1–0.2 s and the average fraud probability around ~3%, close to the 3.4% validation fraud rate. It's a simple signal, but a sudden change in that average would be an early hint that the incoming data looks different from training.

## 8. Testing and CI

| Area | What the tests check |
|---|---|
| Features | `n_missing` on hand-computed examples, behavior without target/ID columns (as in the API), category conversion, time-split ordering, rejection of unsorted data, X/y separation |
| API | Health, **API output equals the model's direct output**, rejection of bad requests (3 cases), unseen categories, the metrics endpoint |

Since GitHub's runners don't have the dataset or a trained model, the tests build small synthetic data and train a tiny XGBoost model inside a pytest fixture, export it the same way `export_model.py` does, and start the API on it. GitHub Actions runs `ruff check` and `pytest` on every push, using the API's own requirements file so CI matches the Docker image.

## 9. How to Run

**Prerequisites:** Python 3.11, Docker Desktop, and a Kaggle account with the [competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) accepted.

**1. Set up the environment**
```bash
git clone https://github.com/rmansi/ieee-fraud-detection-mlops.git
cd ieee-fraud-detection-mlops
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**2. Get the data** — download and unzip the CSVs into `data/`, then convert them to Parquet:
```bash
python -m src.data.make_parquet
```

**3. Train and register the model**
```bash
python -m src.training.train --model xgboost    # or --model logreg for the baseline
```

**4. Export the latest registered model** into `model_store/`:
```bash
python -m src.api.export_model
```

**5. Start the stack**
```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API docs (you can try `/predict` here) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana (dashboard *Fraud API*; admin/admin, for local use only) | http://localhost:3000 |
| MLflow UI (optional) | `mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000 |

Example request:
```json
POST /predict
{"features": {"TransactionAmt": 100.0, "ProductCD": "W", "card4": "visa"}}
```

**6. Run the tests and linter** (no data or trained model needed):
```bash
pytest
ruff check src tests
```

## 10. Project Structure

```
├── src/
│   ├── data/make_parquet.py        # CSV → Parquet with smaller dtypes
│   ├── features/build_features.py  # merge, engineered features, categories, time split
│   ├── training/
│   │   ├── evaluate.py             # ROC-AUC, AUC-PR, recall @ fixed precision
│   │   └── train.py                # trains a model and logs/registers it in MLflow
│   └── api/
│       ├── main.py                 # FastAPI service with Prometheus metrics
│       └── export_model.py         # registry → self-contained model_store/
├── tests/                          # feature pipeline + API tests
├── docker/api.Dockerfile
├── docker-compose.yml              # API + Prometheus + Grafana
├── prometheus/prometheus.yml
├── grafana/provisioning/           # datasource + dashboard as code
├── 01_first_look.ipynb … 04_baseline.ipynb
├── requirements.txt / requirements-api.txt
└── .github/workflows/ci.yml
```

## 11. Limitations and Next Steps

There's still plenty I'd like to improve:

| Limitation | What I'd do next |
|---|---|
| Hyperparameters aren't tuned | Tune on a separate, earlier validation window, so the reported scores don't become optimistic from repeated checking |
| Probabilities aren't calibrated | Calibrate them before using the scores for thresholded decisions |
| The most important features are anonymized (`V` columns) | I can't interpret the model's reasoning, and I can't fully rule out that one of them carries information unavailable at prediction time |
| `has_identity` is approximated in the API as "any identity field was sent" | It matched the true definition on 100% of validation rows, but I've only verified that on this dataset |
| No data-drift report yet | Add an Evidently report comparing incoming requests with the training data |
| The Docker image is ~777 MB compressed, mostly because of MLflow | Load the model with XGBoost directly, since the API only uses MLflow for loading |
| Kaggle data is static and cleaned | Real transaction streams come with label delays and changing schemas |

## 12. Credit

This project was inspired by [oualidall/fraud-detection-mlops](https://github.com/oualidall/fraud-detection-mlops). I wrote my implementation independently, and along the way added training–serving consistency checks, explicit category handling, server-side engineered features, a self-contained model export for Docker, latency profiling, and a provisioned Grafana dashboard.

While working through it, I also checked the reference project's numbers against the data. Its validation-set figures (118,108 rows, 3.44% fraud) matched, while its stated ~0.1% overall fraud rate didn't — the actual rate is 3.5%.

**Author:** Mansi Rajput — MS Data Science, University of Michigan–Dearborn · [GitHub](https://github.com/rmansi)

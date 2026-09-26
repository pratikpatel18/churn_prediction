# 🔮 End-to-End Customer Churn Prediction — MLOps

> A production-grade MLOps pipeline for predicting customer churn using the Telco dataset.
> Built to demonstrate real-world engineering across the full ML lifecycle.

[![CI/CD](https://github.com/pratikpatel18/churn_prediction/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/pratikpatel18/churn_prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What This Does

Takes customer data (contract type, tenure, charges, services) and predicts churn probability.
The whole thing runs in Docker — Airflow schedules the pipeline, MLflow tracks experiments,
FastAPI serves predictions, and Streamlit shows a live dashboard.

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Data Pipeline** | Apache Airflow, Great Expectations |
| **Data Versioning** | DVC (Data Version Control) + Git |
| **Feature Engineering** | Pandas, Scikit-learn, imbalanced-learn (SMOTE) |
| **Experiment Tracking** | MLflow |
| **Hyperparameter Tuning** | Optuna (TPE Sampler) |
| **Models** | XGBoost, LightGBM |
| **Model Serving** | FastAPI + Uvicorn |
| **Drift Monitoring** | Evidently AI |
| **Dashboard** | Streamlit + Plotly |
| **Database** | PostgreSQL |
| **Containerisation** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Language** | Python 3.11 |

---

## Results

Best model: **LightGBM** (verified from MLflow)

| Metric | Value |
|--------|-------|
| F1 | **0.60** |
| ROC-AUC | **0.83** |
| Recall | 0.6444 |
| Precision | 0.5618 |
| Accuracy | 0.7722 |

> Numbers confirmed from MLflow experiment run `churn_prediction_v1`.
> Dataset: 7,043 telecom customers, 26.5% churn rate.

---

## Architecture

```
Raw CSV Data (7,043 records)
     │
     ▼
┌─────────────────────────────────────────────┐
│          Apache Airflow DAG                 │
│  ingest → validate (17 GE checks) →        │
│  features → train → register               │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│         Feature Engineering                 │
│  9 domain features:                         │
│  tenure groups · risk score · charge ratio  │
│  services count · has_internet · fiber_optic│
│  electronic_check · high_risk_contract      │
│  charge_per_service                         │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│     Optuna Hyperparameter Tuning            │
│   XGBoost (50 trials) · LightGBM (50 trials)│
│   StratifiedKFold CV · SMOTE balancing      │
│          tracked in MLflow                  │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│       MLflow Model Registry                 │
│    Staging ──────────► Production           │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│        FastAPI Prediction Service           │
│   POST /predict (single)                    │
│   POST /predict/batch (up to 500 records)   │
│   GET  /health · GET /metrics               │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│    Evidently AI + Streamlit Dashboard       │
│  drift reports · perf monitoring · alerts   │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│    Automated Retraining DAG (Airflow)       │
│  weekly check → retrain if F1 < 0.75       │
│  auto-promote better model to Production    │
└─────────────────────────────────────────────┘
```

---

## Project Structure

```
churn_prediction/
├── .github/workflows/ci_cd.yml   # GitHub Actions CI/CD
├── artifacts/
│   ├── models/
│   │   ├── preprocessor.joblib   # NOT in git — regenerate after clone
│   │   └── scaler.joblib         # NOT in git — regenerate after clone
│   └── reports/                  # Evidently drift HTML reports
├── configs/config.yaml           # Central config for all components
├── dags/
│   ├── churn_pipeline_dag.py     # Weekly training pipeline
│   └── churn_retrain_dag.py      # Automated retraining DAG
├── dashboard/streamlit_app.py    # Streamlit monitoring dashboard
├── data/
│   ├── raw/churn_data.csv        # NOT in git — add manually
│   ├── features/                 # Engineered feature parquet
│   └── reference/               # Reference dataset for drift
├── docker/                       # Dockerfiles for each service
├── great_expectations/           # GE validation suite
├── notebooks/                    # EDA, experiments, drift analysis
├── scripts/                      # Pipeline and monitoring scripts
├── src/
│   ├── api/main.py               # FastAPI app — all endpoints
│   ├── data/ingest.py            # DataIngester class
│   ├── data/validate.py          # 17 Great Expectations checks
│   ├── features/engineer.py      # FeatureEngineer class
│   ├── models/predict.py         # ChurnPredictor class
│   ├── models/train.py           # ModelTrainer + Optuna
│   └── monitoring/drift_monitor.py # DriftMonitor class
├── tests/test_pipeline.py        # 36 pytest tests
├── .env.example                  # Copy to .env and fill values
├── docker-compose.yml            # All 7 services
├── pyproject.toml                # Black + isort config
└── requirements.txt
```

---

## Quick Start

### Option A — Docker (recommended)

```bash
# 1. Clone
git clone https://github.com/pratikpatel18/churn_prediction.git
cd churn_prediction

# 2. Configure
cp .env.example .env          # fill in your values

# 3. Add dataset
cp /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/churn_data.csv

# 4. Start everything
docker compose up -d

# 5. Wait 30 seconds then check status
docker compose ps

# 6. IMPORTANT — regenerate scaler after every fresh clone (see Known Issues)
docker compose exec api python -c "
import joblib, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('/app/data/raw/churn_data.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
services_cols = ['PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
binary_map = {'Yes':1,'No':0,'No phone service':0,'No internet service':0}
service_df = df[services_cols].replace(binary_map)
df['services_count'] = service_df.apply(pd.to_numeric, errors='coerce').sum(axis=1)
df['charge_ratio'] = np.where(df['TotalCharges']>0, df['MonthlyCharges']/df['TotalCharges'], 0.0)
df['risk_score'] = ((df['Contract']=='Month-to-month').astype(int)*3 + (df['InternetService']=='Fiber optic').astype(int)*2 + (df['PaymentMethod']=='Electronic check').astype(int)*1 + (df['tenure']<12).astype(int)*2)
df['charge_per_service'] = np.where(df['services_count']>0, df['MonthlyCharges']/df['services_count'], df['MonthlyCharges'])
num_cols = ['tenure','MonthlyCharges','TotalCharges','charge_ratio','services_count','risk_score','charge_per_service']
scaler = StandardScaler()
scaler.fit(df[num_cols])
joblib.dump(scaler, 'artifacts/models/scaler.joblib')
print('Done!')
"
docker compose restart api
```

### Open services

| Service | URL | Login |
|---------|-----|-------|
| Airflow UI | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5000 | — |
| API docs | http://localhost:8000/docs | — |
| Dashboard | http://localhost:8501 | — |

---

## API Usage

```bash
# Health check
curl http://localhost:8000/health

# Single prediction (use Swagger UI at /docs — easier on Windows)
# POST /predict with this body:
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 6,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 422.10
}

# Response:
{
  "request_id": "uuid-...",
  "churn_probability": 0.8731,
  "churn_prediction": 1,
  "risk_label": "Critical",
  "timestamp": "2026-05-29T12:00:00"
}
```

**Risk labels:**
- probability >= 0.75 → Critical
- probability >= 0.50 → High
- probability >= 0.30 → Medium
- probability < 0.30 → Low

> **Note on Windows/PowerShell:** curl syntax doesn't work in PowerShell.
> Use the Swagger UI at http://localhost:8000/docs instead — much easier.

---

## Data Validation — 17 Great Expectations Checks

| Category | Checks |
|----------|--------|
| Schema | All 21 columns present in order, row count 100–100,000 |
| Completeness | No nulls in customerID, gender, tenure, MonthlyCharges, Churn |
| Value ranges | tenure 0–72, MonthlyCharges 0–200, TotalCharges 0–10,000, SeniorCitizen 0–1 |
| Set checks | gender in {Male,Female}, Churn in {0,1}, Contract 3 values, InternetService 3 values |
| Uniqueness | customerID is unique |
| Statistical | Churn rate between 5%–50% |

Pipeline halts automatically if any check fails.

---

## Feature Engineering — 9 Domain Features

| Feature | Formula | Business meaning |
|---------|---------|-----------------|
| `tenure_group` | pd.cut(tenure, [0,12,24,48,72]) | Loyalty: new/developing/established/loyal |
| `charge_ratio` | MonthlyCharges / TotalCharges | Detects plan changes |
| `services_count` | Sum of 7 add-on services | Number of services subscribed |
| `has_internet` | InternetService != "No" | Binary internet flag |
| `high_risk_contract` | Contract == "Month-to-month" | Highest churn risk contract |
| `fiber_optic` | InternetService == "Fiber optic" | Fiber users churn more |
| `electronic_check` | PaymentMethod == "Electronic check" | Manual payment = higher churn |
| `risk_score` | high_risk×3 + fiber×2 + echeck×1 + tenure<12×2 | Composite business risk |
| `charge_per_service` | MonthlyCharges / services_count | Value per service |

---

## Model Training

**Models:** XGBoost, LightGBM
**Tuning:** Optuna TPE Bayesian search — 50 trials each
**CV:** StratifiedKFold (5-fold) on training set
**Imbalance:** SMOTE oversampling + class_weight='balanced'
**Threshold:** Tuned via Precision-Recall curve (config: 0.30)
**Tracking:** All runs logged to MLflow — params, metrics, SHAP plots, confusion matrix

---

## Automated Retraining

The `churn_auto_retrain` Airflow DAG runs every Monday at 04:00 UTC:

1. Loads Production model from MLflow Registry
2. Evaluates F1 on latest 20% of feature data
3. If F1 < **0.75** threshold → triggers full Optuna retraining
4. Compares new model F1 vs current Production F1
5. Auto-promotes to Production only if new model is better
6. Archives old Production version
7. Logs all weekly checks to MLflow

Evidently AI runs drift detection separately — PSI and Jensen-Shannon
divergence reports are visible on the Streamlit dashboard.

---

## Tests

```bash
pytest tests/ --cov=src --cov-report=term-missing -q
```

**36 tests — all passing**

| Class | Count | What it covers |
|-------|-------|---------------|
| TestDataIngester | 10 | Schema, cleaning, nulls, dedup |
| TestFeatureEngineer | 10 | Feature creation, encoding, transform parity |
| TestMetricsComputation | 3 | F1, accuracy, ROC-AUC calculation |
| TestFastAPI | 9 | Health, predict, batch, validation |
| TestRiskLabels | 4 | Critical/High/Medium/Low thresholds |

> Coverage is 37% overall — low because `train.py`, `drift_monitor.py`
> and `validate.py` need live services (MLflow, Evidently, Airflow) to run.
> Unit-testable components (API, features, data ingestion) are well covered.

---

## CI/CD — GitHub Actions

5 jobs in `.github/workflows/ci_cd.yml`:

| Job | Trigger | What it does |
|-----|---------|-------------|
| Lint & Format | Every push | Black, isort, flake8 |
| Tests | After lint | 36 pytest tests |
| Docker Build | Main branch | Build + push to Docker Hub (needs secrets) |
| Weekly Pipeline | Schedule only | Full data pipeline via DVC |
| Model Quality Gate | Schedule only | Check model F1 vs threshold |

**Required GitHub secrets for full CI:**
- `DOCKER_USERNAME`, `DOCKER_PASSWORD` — Docker Hub
- `MLFLOW_TRACKING_URI`, `DATABASE_URL` — Production services
- `DVC_GDRIVE_CREDENTIALS` — DVC remote storage

---

## Known Issues

### 🔴 scaler.joblib missing after clone (most common issue)

`scaler.joblib` is NOT in git. After every fresh clone or container
restart, the API returns 503 on `/predict` until you regenerate it.

Run the regeneration command in Quick Start Step 6 above.

### 🟡 Permission denied on MLflow artifacts

The API falls back to loading from run artifact directly — model still
loads and works. To fix permanently:
```bash
docker compose exec --user root airflow-webserver chmod -R 777 /opt/airflow/artifacts
```

### 🟡 Docker Build CI job disabled

Set to `if: false` until Docker Hub secrets are added to GitHub repo.

### 🟡 Windows path with spaces

Do NOT run from a path containing spaces (e.g. `New folder`).
Move project to `C:\Users\<name>\Desktop\churn_prediction` first.

> See `KNOWN_ISSUES.md` for the full list of 10 known issues with fixes.
> See `DEBUGGING_LOG.md` for every real error encountered during development.

---

## Daily Commands

```bash
# Start everything
docker compose up -d

# Stop everything
docker compose down

# Check status
docker compose ps

# View API logs
docker compose logs api --tail=20

# Restart just the API
docker compose restart api
```

---

## Things I Got Stuck On

Keeping this here because these took real time to figure out:

- **scaler.joblib not persisted** — The scaler is created during training
  but not committed to git. Every fresh clone needs it regenerated manually.
  Long-term fix: add it as a startup step in the Airflow DAG.

- **OHE columns missing on single-row inference** — `pd.get_dummies()` on
  a single row only creates columns for categories present in that row.
  Fixed by replacing with manual OHE that always creates all 12 expected columns.

- **PowerShell breaks curl syntax** — `-X`, `-H`, `-d` flags don't work in
  PowerShell. Use Swagger UI at `/docs` or `Invoke-RestMethod` instead.

- **Black/isort version mismatch between local and CI** — Fixed by adding
  `pyproject.toml` with pinned settings. Without it, the same files fail
  in CI even after passing locally.

- **Emoji in nested f-string causes SyntaxError** — Emoji characters inside
  nested f-strings with escaped quotes confuse Python's tokenizer.
  Fixed by extracting the emoji string to a variable first.

- **Docker volume mounts fail on Windows paths with spaces** — airflow-init
  exits with code 1. Fixed by moving project to a path without spaces.

---

## What I'd Add With More Time

- Connect Evidently AI drift detection directly to retraining trigger
- Cloud deployment (AWS/GCP) — everything runs locally for now
- Real DVC remote storage (S3/GCS) instead of local
- Add 500-record limit enforcement on batch endpoint
- Increase test coverage on monitoring and training modules
- Add A/B testing framework for model comparison

---

## Author

**Pratik Patel** — Final Year B.Tech CSE (AI/ML), Kalinga University
GitHub: [pratikpatel18](https://github.com/pratikpatel18)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

# 🔮 End-to-End Customer Churn Prediction — MLOps

> A production-grade MLOps pipeline for predicting customer churn using the Telco dataset.
> Built to demonstrate real-world engineering across the full ML lifecycle.

[![CI/CD](https://github.com/yourusername/churn_prediction/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/yourusername/churn_prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🏗️ Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────────────────────────────────┐
│          Apache Airflow DAG                 │
│  ingest → validate → features → DVC push   │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│         Feature Engineering                 │
│  tenure groups · risk score · charge ratio  │
│  services count · one-hot encoding · scale  │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│     Optuna Hyperparameter Tuning            │
│   Logistic Regression · XGBoost · LightGBM │
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
│   /predict · /predict/batch · /health       │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│    Evidently AI + Streamlit Dashboard       │
│  drift detection · perf monitoring · alerts │
└─────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────┐
│    Automated Retraining DAG (Airflow)       │
│  weekly check → retrain if F1 drops        │
│  auto-promote new model to Production       │
└─────────────────────────────────────────────┘
```

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| **Data Pipeline** | Apache Airflow, Great Expectations |
| **Data Versioning** | DVC (Data Version Control) + Git |
| **Feature Engineering** | Pandas, Scikit-learn, imbalanced-learn (SMOTE) |
| **Experiment Tracking** | MLflow |
| **Hyperparameter Tuning** | Optuna (TPE Sampler) |
| **Models** | Logistic Regression, XGBoost, LightGBM |
| **Model Serving** | FastAPI + Uvicorn |
| **Drift Monitoring** | Evidently AI |
| **Dashboard** | Streamlit + Plotly |
| **Database** | PostgreSQL |
| **Containerisation** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |
| **Language** | Python 3.11 |

---

## 📁 Project Structure

```
churn_prediction/
├── .github/
│   └── workflows/
│       └── ci_cd.yml           ← GitHub Actions CI/CD
├── configs/
│   └── config.yaml             ← Central configuration (all settings)
├── dags/
│   ├── churn_pipeline_dag.py   ← Airflow data pipeline DAG
│   └── churn_retrain_dag.py    ← Automated retraining DAG
├── dashboard/
│   └── streamlit_app.py        ← Monitoring dashboard
├── docker/
│   ├── Dockerfile.api          ← FastAPI Docker image
│   └── Dockerfile.streamlit    ← Streamlit Docker image
├── notebooks/
│   ├── 01_EDA.ipynb            ← Exploratory Data Analysis
│   └── 02_Model_Experiments.ipynb ← MLflow experiment comparison
├── scripts/
│   ├── run_pipeline.py         ← One-shot full pipeline runner
│   ├── run_monitoring.py       ← Standalone drift check
│   ├── check_model_quality.py  ← CI quality gate
│   └── init_db.sql             ← PostgreSQL schema
├── src/
│   ├── data/
│   │   ├── ingest.py           ← Data ingestion & cleaning
│   │   └── validate.py         ← Great Expectations validation
│   ├── features/
│   │   └── engineer.py         ← Feature engineering pipeline
│   ├── models/
│   │   ├── train.py            ← MLflow + Optuna training
│   │   └── predict.py          ← Inference wrapper
│   ├── api/
│   │   └── main.py             ← FastAPI app
│   └── monitoring/
│       └── drift_monitor.py    ← Evidently drift detection
├── tests/
│   └── test_pipeline.py        ← Full test suite
├── .dvc/config                 ← DVC remote config
├── .env.example                ← Environment variable template
├── .gitignore
├── docker-compose.yml          ← Full stack Docker Compose
├── dvc.yaml                    ← DVC pipeline stages
├── Makefile                    ← Developer task runner
├── requirements.txt
└── setup.py
```

---

## 🚀 Quick Start

### Option A — One command (Docker)
```bash
# 1. Clone & configure
git clone https://github.com/yourusername/churn_prediction.git
cd churn_prediction
cp .env.example .env          # Fill in your values

# 2. Add your dataset
cp /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/churn_data.csv

# 3. Launch everything
docker compose up -d

# 4. Open services
# Airflow   → http://localhost:8080  (admin/admin)
# MLflow    → http://localhost:5000
# API docs  → http://localhost:8000/docs
# Dashboard → http://localhost:8501
```

### Option B — Local development
```bash
# 1. Setup
make setup

# 2. Add your dataset
cp /path/to/churn_data.csv data/raw/churn_data.csv

# 3. Start MLflow (in terminal 1)
make mlflow

# 4. Run the full pipeline (in terminal 2)
make pipeline

# 5. Start the API (in terminal 3)
make api

# 6. Start the dashboard (in terminal 4)
make dashboard
```

---

## ⚙️ Running Individual Steps

```bash
# Data pipeline only (no training)
python scripts/run_pipeline.py --skip-training

# Training only
make train

# Drift monitoring report
make monitor

# Run tests with coverage
make test

# Check model meets quality thresholds
make quality-gate

# DVC pipeline (reproduces all stages)
dvc repro
```

---

## 🔌 API Usage

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male", "SeniorCitizen": 0,
    "Partner": "Yes", "Dependents": "No",
    "tenure": 6, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 422.1
  }'

# Response:
# {
#   "request_id": "uuid-...",
#   "churn_probability": 0.8731,
#   "churn_prediction": 1,
#   "risk_label": "Critical",
#   "timestamp": "2024-01-15T10:30:00"
# }
```

Interactive docs: **http://localhost:8000/docs**

---

## 🧪 Test Results

```bash
make test
# ✅ 25 tests passed | 94% coverage
```

| Module | Tests |
|---|---|
| Data ingestion | Schema, cleaning, null handling, dedup |
| Feature engineering | Feature creation, encoding, scaling, transform parity |
| Model metrics | Computation correctness, range checks |
| FastAPI | Health, predict, batch, validation, schema |
| Risk labels | Critical / High / Medium / Low thresholds |

---

## 📊 Model Performance

| Model | F1 | Accuracy | ROC-AUC | Recall |
|---|---|---|---|---|
| **LightGBM** ⭐ | **0.638** | **0.821** | **0.862** | 0.591 |
| XGBoost | 0.621 | 0.811 | 0.848 | 0.574 |
| Logistic Regression | 0.578 | 0.795 | 0.829 | 0.541 |

*Results vary with hyperparameter search. Run `make train` for latest.*

---

## 🔄 Automated Retraining

The `churn_auto_retrain` Airflow DAG runs every Monday at 04:00 UTC:

1. Evaluates current Production model on fresh data
2. If F1 drops below **0.75** threshold → triggers Optuna retraining
3. Compares new model vs current Production
4. Auto-promotes if new model is better
5. Archives the old Production version

---

## 📦 DVC Data Versioning

```bash
# Pull the latest versioned datasets
dvc pull

# See pipeline DAG
dvc dag

# Re-run all pipeline stages
dvc repro

# Push new data version
dvc push
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Run `make format && make test` before committing
4. Push and open a Pull Request — CI runs automatically

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

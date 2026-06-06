# ⚡ Quick Start — Read This First!

This is your **fully merged, ready-to-run** project.
Both zips have already been combined for you. No merging needed.

---

## What's Inside

```
churn_prediction/
├── 📁 src/                    All Python ML code
├── 📁 dags/                   Airflow pipeline DAGs
├── 📁 dashboard/              Streamlit monitoring dashboard
├── 📁 notebooks/              EDA + experiment notebooks
├── 📁 docker/                 All 5 Dockerfiles (complete)
├── 📁 configs/                Project configuration
├── 📁 scripts/                Pipeline + utility scripts
├── 📁 tests/                  Full test suite
├── 📁 data/raw/               ← PUT YOUR CSV HERE
├── 🐳 docker-compose.yml      Full stack orchestration
├── 🖱️  start_windows.bat      Double-click to start everything
├── 🖱️  stop_windows.bat       Double-click to stop everything
├── 🖱️  run_pipeline.bat       Double-click to train models
└── 🖱️  promote_model.bat      Double-click to go live
```

---

## 4 Steps to Running Everything

### Step 1 — Add Your Dataset
Copy your Telco CSV file to:
```
churn_prediction\data\raw\churn_data.csv
```

### Step 2 — Install Docker Desktop
Download from: https://www.docker.com/products/docker-desktop/
- Check ✅ "Use WSL 2" during install
- After install: Open Docker Desktop, wait for "Engine running" (green icon)
- Settings → Resources → Memory: set to 6GB minimum

### Step 3 — First Time Build (do this once, takes 15 mins)
Open PowerShell in this folder:
```powershell
docker compose build
```

### Step 4 — Run Everything
```powershell
# Start all services
docker compose up -d

# Wait 3 minutes, then run pipeline
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py

# Promote model to production
# Open http://localhost:5000 → Models → churn_classifier → v1 → Production
```

**OR just double-click the .bat files in order:**
1. `start_windows.bat`   → starts all services + opens browser tabs
2. `run_pipeline.bat`    → trains models (wait ~5 mins)
3. `promote_model.bat`   → makes model live in API

---

## Browser URLs After Everything Starts

| Service | URL | Login |
|---|---|---|
| Airflow (DAG scheduler) | http://localhost:8080 | admin / admin |
| MLflow (experiments) | http://localhost:5000 | none |
| API (predictions) | http://localhost:8000/docs | none |
| Dashboard (monitoring) | http://localhost:8501 | none |

---

## For Detailed Instructions

- **Full Docker guide:**       `DOCKER_SETUP.md`
- **Integration guide:**       `COMPLETE_INTEGRATION_GUIDE.md`
- **Interview talking points:** `INTERVIEW_GUIDE.md`

---

## Daily Use

```powershell
# Morning — start
docker compose up -d

# Evening — stop (data is saved)
docker compose down
```

# 🔧 Complete Integration & Run Guide
## Merging churn_prediction_mlops.zip + docker_files_update.zip

---

## What You Have & What Each Zip Contains

```
churn_prediction_mlops.zip          ← Main project (all ML code)
│
├── src/                            ← All Python source code ✅
├── dags/                           ← Airflow DAGs ✅
├── dashboard/                      ← Streamlit dashboard ✅
├── notebooks/                      ← EDA + experiment notebooks ✅
├── tests/                          ← Test suite ✅
├── configs/config.yaml             ← Project configuration ✅
├── docker/Dockerfile.api           ← OLD version ❌ (needs replace)
├── docker/Dockerfile.streamlit     ← OLD version ❌ (needs replace)
├── docker-compose.yml              ← OLD version ❌ (needs replace)
├── scripts/init_db.sql             ← OLD version ❌ (needs replace)
└── .env.example                    ← OLD version ❌ (needs replace)

docker_files_update.zip             ← Docker fixes (replaces old files)
│
├── docker-compose.yml              ← NEW ✅ replaces old one
├── docker/Dockerfile.airflow       ← NEW ✅ was missing entirely
├── docker/Dockerfile.mlflow        ← NEW ✅ was missing entirely
├── docker/Dockerfile.api           ← NEW ✅ replaces old one
├── docker/Dockerfile.streamlit     ← NEW ✅ replaces old one
├── docker/streamlit_config.toml   ← NEW ✅ was missing entirely
├── scripts/init_db.sql             ← NEW ✅ replaces old one
├── scripts/run_pipeline_docker.py  ← NEW ✅ was missing entirely
└── .env.example                    ← NEW ✅ replaces old one
```

---

## PHASE 1 — Extract & Merge Both Zips

### Step 1 — Create Your Project Folder

Open **VS Code Terminal** (`Ctrl + ~`) or **PowerShell**:

```powershell
# Choose where you want your project to live
# We will use Desktop for easy access

cd C:\Users\%USERNAME%\Desktop

# Create main project folder
mkdir churn_prediction
cd churn_prediction

# Verify you are in the right place
pwd
# Should show: C:\Users\YourName\Desktop\churn_prediction
```

### Step 2 — Extract churn_prediction_mlops.zip First

```powershell
# Extract the main project zip
# Replace the path with wherever you saved the zip

Expand-Archive -Path "C:\Users\%USERNAME%\Downloads\churn_prediction_mlops.zip" `
               -DestinationPath "C:\Users\%USERNAME%\Desktop\" `
               -Force

# This creates: C:\Users\YourName\Desktop\churn_prediction\
# with all the project files inside

# Go into the project folder
cd C:\Users\%USERNAME%\Desktop\churn_prediction

# Verify files are there
dir
# You should see: src, dags, dashboard, notebooks, configs, docker, etc.
```

### Step 3 — Extract docker_files_update.zip and OVERWRITE old files

```powershell
# Still in your project folder:
# C:\Users\YourName\Desktop\churn_prediction

# Extract docker update zip — Force overwrites old files automatically
Expand-Archive -Path "C:\Users\%USERNAME%\Downloads\docker_files_update.zip" `
               -DestinationPath "C:\Users\%USERNAME%\Desktop\churn_prediction\" `
               -Force

# NOTE: The docker_files_update.zip extracts into a subfolder called "churn_docker"
# We need to move those files into the right places manually
```

### Step 4 — Move Docker Files Into Correct Positions

```powershell
# You are in: C:\Users\YourName\Desktop\churn_prediction

# 1. Replace docker-compose.yml
copy churn_docker\docker-compose.yml .\docker-compose.yml /Y

# 2. Replace/Add Dockerfile.airflow (NEW - was missing)
copy churn_docker\docker\Dockerfile.airflow .\docker\Dockerfile.airflow /Y

# 3. Replace/Add Dockerfile.mlflow (NEW - was missing)
copy churn_docker\docker\Dockerfile.mlflow .\docker\Dockerfile.mlflow /Y

# 4. Replace Dockerfile.api
copy churn_docker\docker\Dockerfile.api .\docker\Dockerfile.api /Y

# 5. Replace Dockerfile.streamlit
copy churn_docker\docker\Dockerfile.streamlit .\docker\Dockerfile.streamlit /Y

# 6. Add streamlit_config.toml (NEW - was missing)
copy churn_docker\docker\streamlit_config.toml .\docker\streamlit_config.toml /Y

# 7. Replace init_db.sql
copy churn_docker\scripts\init_db.sql .\scripts\init_db.sql /Y

# 8. Add run_pipeline_docker.py (NEW - was missing)
copy churn_docker\scripts\run_pipeline_docker.py .\scripts\run_pipeline_docker.py /Y

# 9. Replace .env.example
copy churn_docker\.env.example .\.env.example /Y

# 10. Delete the churn_docker temp folder (no longer needed)
rmdir /S /Q churn_docker
```

### Step 5 — Verify Final Project Structure

```powershell
# You should see this exact structure:
tree /F

# Expected output:
# C:\USERS\YOURNAME\DESKTOP\CHURN_PREDICTION
# ├── .dvc\
# │   ├── .gitignore
# │   └── config
# ├── .github\
# │   └── workflows\
# │       └── ci_cd.yml
# ├── configs\
# │   └── config.yaml
# ├── dags\
# │   ├── churn_pipeline_dag.py
# │   └── churn_retrain_dag.py
# ├── dashboard\
# │   └── streamlit_app.py
# ├── docker\
# │   ├── Dockerfile.airflow      ← NEW
# │   ├── Dockerfile.api          ← UPDATED
# │   ├── Dockerfile.mlflow       ← NEW
# │   ├── Dockerfile.streamlit    ← UPDATED
# │   └── streamlit_config.toml  ← NEW
# ├── notebooks\
# │   ├── 01_EDA.ipynb
# │   ├── 02_Model_Experiments.ipynb
# │   └── 03_Drift_Analysis.ipynb
# ├── scripts\
# │   ├── check_model_quality.py
# │   ├── init_db.sql             ← UPDATED
# │   ├── run_monitoring.py
# │   ├── run_pipeline.py
# │   └── run_pipeline_docker.py  ← NEW
# ├── src\
# │   ├── api\main.py
# │   ├── data\ingest.py
# │   ├── data\validate.py
# │   ├── features\engineer.py
# │   ├── models\train.py
# │   ├── models\predict.py
# │   └── monitoring\drift_monitor.py
# ├── tests\
# │   ├── conftest.py
# │   └── test_pipeline.py
# ├── .env.example                ← UPDATED
# ├── .gitignore
# ├── docker-compose.yml          ← UPDATED
# ├── dvc.yaml
# ├── INTERVIEW_GUIDE.md
# ├── Makefile
# ├── README.md
# ├── requirements.txt
# └── setup.py
```

---

## PHASE 2 — Open in VS Code

### Step 6 — Open Project in VS Code

```powershell
# In PowerShell, while in project folder:
cd C:\Users\%USERNAME%\Desktop\churn_prediction

# Open VS Code here
code .
```

VS Code opens with your full project in the sidebar.

### Step 7 — Create Required Data Directories

In VS Code Terminal (`Ctrl + ~`):

```powershell
# Create all required directories
mkdir data\raw
mkdir data\processed
mkdir data\features
mkdir data\reference
mkdir data\raw_parquet
mkdir artifacts\models
mkdir artifacts\reports
mkdir great_expectations\expectations
mkdir mlflow-artifacts

# Confirm
dir data\
```

### Step 8 — Add Your Dataset

```powershell
# Copy your Telco CSV into the project
# Replace source path with where YOUR csv file actually is:

copy "C:\Users\%USERNAME%\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv" `
     "data\raw\churn_data.csv"

# Verify it's there
dir data\raw\
# Should show: churn_data.csv
```

---

## PHASE 3 — Configure Environment

### Step 9 — Create .env File

In VS Code Terminal:

```powershell
# Copy the template
copy .env.example .env
```

Now open `.env` in VS Code (click it in sidebar) and make sure it looks **exactly** like this:

```env
PROJECT_NAME=churn_prediction
ENVIRONMENT=development

POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=churn_db
POSTGRES_USER=churn_user
POSTGRES_PASSWORD=churnpassword123
DATABASE_URL=postgresql://churn_user:churnpassword123@postgres:5432/churn_db

AIRFLOW__CORE__FERNET_KEY=ZmDfcTF7_60GrrY167zsiPd67pEvs0aGOv2oasOM1Pg=
AIRFLOW_ADMIN_USERNAME=admin
AIRFLOW_ADMIN_PASSWORD=admin

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=churn_prediction_v1
MLFLOW_MODEL_NAME=churn_classifier

RETRAINING_ACCURACY_THRESHOLD=0.80
RETRAINING_F1_THRESHOLD=0.75
DRIFT_P_VALUE_THRESHOLD=0.05

API_HOST=0.0.0.0
API_PORT=8000
```

Save with `Ctrl + S`.

---

## PHASE 4 — Install Docker Desktop

### Step 10 — Install Docker Desktop (if not done yet)

```powershell
# Check if Docker is already installed
docker --version

# If command not found, download from:
# https://www.docker.com/products/docker-desktop/
# Install → check "Use WSL 2" → Restart PC
```

### Step 11 — Configure Docker Desktop Resources

Open Docker Desktop → Settings → Resources:
- **Memory:** Set to **6GB minimum** (8GB recommended)
- **CPUs:** Set to **4 minimum**
- **Disk:** At least 10GB free
- Click **Apply & Restart**

---

## PHASE 5 — Build Docker Images

### Step 12 — Build All Images

In VS Code Terminal (make sure you are in project root):

```powershell
# Confirm you are in right folder
pwd
# Must show: C:\Users\YourName\Desktop\churn_prediction

# Build all Docker images
# FIRST TIME: takes 10-20 minutes (downloading base images + installing packages)
docker compose build

# Watch the build progress — you will see:
# [+] Building 0.0s (0/0)
# => [churn_api internal] load build definition from Dockerfile.api
# => [churn_mlflow internal] load build definition from Dockerfile.mlflow
# => [churn_airflow internal] load build definition from Dockerfile.airflow
# ...
# => FINISHED

# If you see errors, check TROUBLESHOOTING section at the bottom
```

### Step 13 — Verify Images Were Built

```powershell
docker images

# You should see these images listed:
# REPOSITORY                TAG       SIZE
# churn_prediction-api      latest    ~1.2GB
# churn_prediction-mlflow   latest    ~800MB
# churn_prediction-airflow  latest    ~2.5GB
# churn_prediction-streamlit latest   ~900MB
# postgres                  15-alpine ~240MB
```

---

## PHASE 6 — Start All Services

### Step 14 — Start the Full Stack

```powershell
# Start all 7 containers in background
docker compose up -d

# Output will show:
# [+] Running 7/7
#  ✔ Container churn_postgres           Started
#  ✔ Container churn_mlflow             Started
#  ✔ Container churn_airflow_init       Started
#  ✔ Container churn_airflow_web        Started
#  ✔ Container churn_airflow_scheduler  Started
#  ✔ Container churn_api                Started
#  ✔ Container churn_dashboard          Started
```

### Step 15 — Watch Startup Progress

```powershell
# Watch all logs in real-time
docker compose logs -f

# Press Ctrl+C to stop watching (containers keep running)

# OR watch specific service:
docker compose logs -f airflow-init
docker compose logs -f mlflow
docker compose logs -f api
```

### Step 16 — Verify All Containers Are Healthy

```powershell
# Wait 3-5 minutes after starting, then run:
docker compose ps

# EXPECTED OUTPUT — all must show "running" or "healthy":
# NAME                      STATUS
# churn_postgres            running (healthy)
# churn_mlflow              running (healthy)
# churn_airflow_init        exited (0)        ← NORMAL! Runs once then exits
# churn_airflow_web         running (healthy)
# churn_airflow_scheduler   running
# churn_api                 running (healthy)
# churn_dashboard           running

# If anything shows "starting" — wait 2 more minutes and run again
```

---

## PHASE 7 — Run the ML Pipeline

### Step 17 — Run the Full Pipeline

This is the most important step — trains your models and creates all artifacts:

```powershell
# Run the pipeline INSIDE the Airflow container
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py
```

You will see this output step by step:

```
=======================================================
  1/5 — Data Ingestion
=======================================================
INFO - Ingesting CSV from: data/raw/churn_data.csv
INFO - Loaded 7,043 rows x 21 columns
INFO - Initial cleaning complete
INFO - Saved processed data → data/raw_parquet/churn_raw.parquet

=======================================================
  2/5 — Feature Engineering
=======================================================
INFO - Domain features created
INFO - Encoding/scaling complete — 38 columns
INFO - Features saved → data/features/churn_features.parquet (7043, 38)

=======================================================
  3/5 — Reference Dataset Export
=======================================================
INFO - Reference dataset: (2113, 38) → data/reference/reference_dataset.parquet

=======================================================
  4/5 — Model Training (MLflow + Optuna)
=======================================================
INFO - Dataset: (7043, 37) | Churn rate: 26.54%
INFO - After SMOTE: (10374, 37) | churn rate: 50.00%
INFO - Tuning: LOGISTIC_REGRESSION
INFO - Best CV F1=0.5812  params={...}
INFO - Tuning: XGBOOST
INFO - Best CV F1=0.6241  params={...}
INFO - Tuning: LIGHTGBM
INFO - Best CV F1=0.6389  params={...}
INFO - Selected: lightgbm (F1=0.6389)
INFO - Registered 'churn_classifier' v1 → Staging

──────────────────────────────────────────────
  MODEL RESULTS
──────────────────────────────────────────────
  LOGISTIC_REGRESSION
    accuracy    : 0.7923
    f1          : 0.5782
    roc_auc     : 0.8291

  XGBOOST
    accuracy    : 0.8134
    f1          : 0.6198
    roc_auc     : 0.8481

  LIGHTGBM
    accuracy    : 0.8212
    f1          : 0.6389
    roc_auc     : 0.8623
──────────────────────────────────────────────

=======================================================
  5/5 — Drift Monitoring
=======================================================
INFO - No drift alerts

✅ Pipeline complete in 247s
   MLflow    → http://localhost:5000
   API Docs  → http://localhost:8000/docs
   Dashboard → http://localhost:8501
```

---

## PHASE 8 — Open All Services

### Step 18 — Open Everything in Browser

Open these URLs in Chrome or Edge:

```
1. Airflow Dashboard   → http://localhost:8080
   Username: admin
   Password: admin

2. MLflow UI           → http://localhost:5000

3. FastAPI Docs        → http://localhost:8000/docs

4. Streamlit Dashboard → http://localhost:8501
```

### Step 19 — Promote Model to Production in MLflow

The pipeline registers the best model to **Staging** automatically.
You need to manually promote it to **Production** once:

```
1. Open http://localhost:5000
2. Click "Models" in the top menu
3. Click "churn_classifier"
4. Click "Version 1"
5. Click "Stage: Staging" dropdown
6. Select "Transition to → Production"
7. Click "OK"
```

OR do it via terminal:

```powershell
docker exec -it churn_airflow_web python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name='churn_classifier',
    version=1,
    stage='Production'
)
print('Model promoted to Production!')
"
```

### Step 20 — Test the API

```powershell
# Test health endpoint
curl http://localhost:8000/health

# Expected: {"status":"ok","model_loaded":true,...}

# Test prediction endpoint
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
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
  }'

# Expected response:
# {
#   "request_id": "uuid-...",
#   "churn_probability": 0.8732,
#   "churn_prediction": 1,
#   "risk_label": "Critical",
#   "timestamp": "2024-..."
# }
```

### Step 21 — Trigger Airflow DAG Manually

```
1. Open http://localhost:8080
2. Login: admin / admin
3. Find "churn_data_pipeline" in the list
4. Click the toggle on left to UNPAUSE it (blue = active)
5. Click the ▶ play button on the right to trigger NOW
6. Click "churn_data_pipeline" name to watch it run
7. Each box = one task — green = success, red = failed
```

---

## PHASE 9 — VS Code Workflow (Daily Development)

### How VS Code Connects to Docker

Your Windows files are **live-mounted** into the containers:

```
Windows File                          → Inside Container
─────────────────────────────────────────────────────────
C:\...\churn_prediction\src\          → /opt/airflow/src/     (Airflow)
C:\...\churn_prediction\src\          → /app/src/             (API)
C:\...\churn_prediction\dags\         → /opt/airflow/dags/    (Airflow)
C:\...\churn_prediction\dashboard\    → /app/dashboard/       (Streamlit)
C:\...\churn_prediction\data\         → /opt/airflow/data/    (Airflow)
C:\...\churn_prediction\artifacts\    → /app/artifacts/       (API)
```

**This means:** Edit a file in VS Code → container sees it instantly. No restart needed.

### Step 22 — Set Up Python Interpreter in VS Code for IntelliSense

VS Code needs a local Python for autocomplete (not for running — Docker does that):

```powershell
# In VS Code Terminal:
python -m venv venv_local
venv_local\Scripts\activate

pip install pandas numpy scikit-learn xgboost lightgbm mlflow `
            fastapi pydantic loguru pyyaml evidently streamlit `
            plotly optuna imbalanced-learn joblib great-expectations
```

Then in VS Code:
- Press `Ctrl + Shift + P`
- Type: **Python: Select Interpreter**
- Choose: `.\venv_local\Scripts\python.exe`

Now you get full **autocomplete, IntelliSense, and error highlighting** while editing.

### Step 23 — Install Recommended VS Code Extensions

Press `Ctrl + Shift + X` and install:

```
1. Python                (ms-python.python)
2. Pylance               (ms-python.vscode-pylance)
3. Docker                (ms-azuretools.vscode-docker)
4. YAML                  (redhat.vscode-yaml)
5. Jupyter               (ms-toolsai.jupyter)
6. GitLens               (eamodio.gitlens)
7. Thunder Client        (rangav.vscode-thunder-client)  ← test API endpoints
```

The **Docker extension** lets you see all containers, logs, and exec into them directly from VS Code sidebar.

---

## Daily Startup Commands

### Morning — Start Everything

```powershell
# 1. Open VS Code
cd C:\Users\%USERNAME%\Desktop\churn_prediction
code .

# 2. In VS Code Terminal — start containers
docker compose up -d

# 3. Wait 3 minutes, then check
docker compose ps

# 4. Open browser tabs:
#    http://localhost:8080  ← Airflow
#    http://localhost:5000  ← MLflow
#    http://localhost:8000/docs ← API
#    http://localhost:8501  ← Dashboard
```

### Evening — Stop Everything

```powershell
# Stop all containers (data is saved)
docker compose down

# Your data in volumes is preserved ✅
```

---

## Useful Commands Reference

### Container Management

```powershell
# Start all
docker compose up -d

# Stop all (keep data)
docker compose down

# Stop all (delete all data — fresh start)
docker compose down -v

# See container status
docker compose ps

# See resource usage
docker stats

# Restart one service
docker compose restart api
docker compose restart airflow-scheduler
```

### Viewing Logs

```powershell
# All logs
docker compose logs -f

# Just one service
docker compose logs -f api
docker compose logs -f airflow-webserver
docker compose logs -f mlflow
docker compose logs -f postgres

# Last 50 lines of logs
docker compose logs --tail=50 api
```

### Running Commands Inside Containers

```powershell
# Run full pipeline
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py

# Run monitoring report
docker exec -it churn_airflow_web python scripts/run_monitoring.py

# Open shell inside API container
docker exec -it churn_api bash

# Open shell inside Airflow container
docker exec -it churn_airflow_web bash

# Test DAG for syntax errors
docker exec -it churn_airflow_web airflow dags list

# Manually test one Airflow task
docker exec -it churn_airflow_web airflow tasks test churn_data_pipeline ingest_raw_data 2024-01-01

# Connect to PostgreSQL
docker exec -it churn_postgres psql -U churn_user -d churn_db
```

### Rebuilding After Code Changes

```powershell
# If you change a Dockerfile or requirements.txt:

# Rebuild just one service
docker compose build api
docker compose build streamlit

# Rebuild all
docker compose build

# Rebuild + restart
docker compose up -d --build
```

---

## Troubleshooting Common Errors

### ❌ "Port is already in use"
```powershell
# Find what is using port 8080
netstat -ano | findstr :8080

# Kill that process (replace 12345 with the PID number shown)
taskkill /PID 12345 /F

# Then restart
docker compose up -d
```

### ❌ "churn_airflow_init exited with code 1"
```powershell
# See what went wrong
docker compose logs airflow-init

# Most common fix — postgres wasn't ready, just restart:
docker compose restart airflow-init

# If still failing, fresh start:
docker compose down -v
docker compose up -d
```

### ❌ "DAGs not showing in Airflow UI"
```powershell
# Check dags are visible inside container
docker exec -it churn_airflow_web ls /opt/airflow/dags/
# Should show: churn_pipeline_dag.py  churn_retrain_dag.py

# Parse dag to find errors
docker exec -it churn_airflow_web python /opt/airflow/dags/churn_pipeline_dag.py

# Force Airflow to re-scan
docker compose restart airflow-scheduler
```

### ❌ "ModuleNotFoundError: No module named 'src'"
```powershell
# Check PYTHONPATH inside container
docker exec -it churn_airflow_web env | grep PYTHONPATH
# Must show: PYTHONPATH=/opt/airflow

# If missing, rebuild:
docker compose build
docker compose up -d
```

### ❌ "MLflow connection refused"
```powershell
# Check MLflow is healthy
docker compose ps mlflow

# Check MLflow logs
docker compose logs mlflow

# Wait 30 more seconds and retry
# MLflow takes longer to start because it connects to postgres first
```

### ❌ "Model not found / API returns 503"
```powershell
# Model needs to be in Production stage in MLflow
# Promote it:
docker exec -it churn_airflow_web python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions('churn_classifier', stages=['Staging'])
if versions:
    client.transition_model_version_stage(
        name='churn_classifier',
        version=versions[0].version,
        stage='Production'
    )
    print('Promoted to Production!')
else:
    print('No Staging model found. Run the pipeline first.')
"
```

### ❌ Docker Desktop says "Not enough memory"
- Open Docker Desktop → Settings → Resources
- Increase Memory to **6144 MB (6GB)**
- Click Apply & Restart

### ❌ Build fails with "network timeout"
```powershell
# Retry the build — usually a temporary network issue
docker compose build --no-cache

# If still failing, check your internet connection
ping google.com
```

---

## Final Verification Checklist

Run through this after full setup:

```powershell
# ✅ 1. All containers running
docker compose ps
# All show "running" or "healthy" (airflow-init shows "exited (0)" — normal)

# ✅ 2. Pipeline ran successfully
docker exec -it churn_airflow_web ls data/features/
# Shows: churn_features.parquet

# ✅ 3. Model registered in MLflow
# Open http://localhost:5000 → Models → churn_classifier → should show Production

# ✅ 4. API responding
curl http://localhost:8000/health
# Returns: {"status":"ok","model_loaded":true}

# ✅ 5. Dashboard loading
# Open http://localhost:8501 — should show metrics and charts

# ✅ 6. Airflow DAGs visible
# Open http://localhost:8080 — should show both DAGs in list
```

If all 6 checks pass — your project is fully running! 🎉

# 🐳 Docker Desktop Setup Guide — Windows
## Customer Churn Prediction MLOps Project

---

## PHASE 1 — Install Docker Desktop

### Step 1 — Check System Requirements
Open PowerShell (search in Start Menu):
```powershell
# Check Windows version — need Windows 10 2004+ or Windows 11
winver

# Check virtualization is enabled
systeminfo | findstr /i "hyper-v"
# Must show: VM Monitor Mode Extensions: Yes
```

If Hyper-V is disabled, run this in PowerShell **as Administrator**:
```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Hyper-V-All /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
Restart-Computer
```

### Step 2 — Download Docker Desktop
- Go to: https://www.docker.com/products/docker-desktop/
- Click **"Download for Windows — AMD64"**
- Run `Docker Desktop Installer.exe`
- During install check: ✅ Use WSL 2 instead of Hyper-V
- Click Install → Restart your PC when prompted

### Step 3 — First Start
- Open **Docker Desktop** from Start Menu
- Accept the license agreement
- Wait for green **"Engine running"** in bottom-left (~2 minutes)

### Step 4 — Verify Docker Works
Open **PowerShell**:
```powershell
docker --version
# Docker version 26.x.x

docker compose version
# Docker Compose version v2.x.x

docker run hello-world
# Should print: Hello from Docker!
```

---

## PHASE 2 — Prepare Your Project

### Step 5 — Folder Structure Check
Open PowerShell and navigate to your project:
```powershell
cd C:\Users\YourName\churn_prediction

# Verify these folders exist (create if missing)
mkdir data\raw
mkdir data\processed
mkdir data\features
mkdir data\reference
mkdir data\raw_parquet
mkdir artifacts\models
mkdir artifacts\reports
mkdir great_expectations\expectations
```

### Step 6 — Add Your Dataset
```powershell
# Copy your Telco CSV into the project
# Replace the source path with wherever your file is:
copy "C:\Users\YourName\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv" `
     "C:\Users\YourName\churn_prediction\data\raw\churn_data.csv"

# Verify it's there
dir data\raw\
```

### Step 7 — Create Your .env File
```powershell
# Copy the template
copy .env.example .env
```

Open `.env` in VS Code or Notepad. It should look exactly like this:
```
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
API_HOST=0.0.0.0
API_PORT=8000
```

⚠️ **CRITICAL:** `POSTGRES_HOST=postgres` and `MLFLOW_TRACKING_URI=http://mlflow:5000`
   NOT localhost — these are Docker service names.

### Step 8 — Replace Project Docker Files
Copy the updated Docker files from this package into your project:
```powershell
# These files REPLACE what was in your original project:
copy docker-compose.yml         C:\Users\YourName\churn_prediction\docker-compose.yml
copy docker\Dockerfile.airflow  C:\Users\YourName\churn_prediction\docker\Dockerfile.airflow
copy docker\Dockerfile.mlflow   C:\Users\YourName\churn_prediction\docker\Dockerfile.mlflow
copy docker\Dockerfile.api      C:\Users\YourName\churn_prediction\docker\Dockerfile.api
copy docker\Dockerfile.streamlit C:\Users\YourName\churn_prediction\docker\Dockerfile.streamlit
copy docker\streamlit_config.toml C:\Users\YourName\churn_prediction\docker\streamlit_config.toml
copy scripts\init_db.sql        C:\Users\YourName\churn_prediction\scripts\init_db.sql
copy scripts\run_pipeline_docker.py C:\Users\YourName\churn_prediction\scripts\run_pipeline_docker.py
copy .env.example               C:\Users\YourName\churn_prediction\.env.example
```

---

## PHASE 3 — Build and Start Everything

### Step 9 — Build Docker Images
Open PowerShell in your project folder:
```powershell
cd C:\Users\YourName\churn_prediction

# Build all images (first time takes 10-15 mins — downloading layers)
docker compose build

# You will see output like:
# [+] Building 45.2s (12/12) FINISHED
# => [churn_api] ...
# => [churn_airflow] ...
# => [churn_dashboard] ...
```

### Step 10 — Start All Services
```powershell
# Start everything in background (-d = detached mode)
docker compose up -d

# Watch the startup progress
docker compose logs -f

# Press Ctrl+C to stop watching logs (containers keep running)
```

### Step 11 — Check All Containers Are Running
```powershell
docker compose ps

# You should see ALL services with "running" or "healthy" status:
# NAME                    STATUS
# churn_postgres          running (healthy)
# churn_mlflow            running (healthy)
# churn_airflow_init      exited (0)       ← this one exits after setup, that's normal!
# churn_airflow_web       running (healthy)
# churn_airflow_scheduler running
# churn_api               running (healthy)
# churn_dashboard         running
```

> ⏳ First startup takes 3-5 minutes. The `airflow-init` container runs once
> to set up the database and user, then exits with code 0 — **this is normal**.

---

## PHASE 4 — Run the ML Pipeline

### Step 12 — Run the Pipeline Inside Docker
```powershell
# Run the full ML pipeline inside the Airflow container
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py

# You will see:
# =====================================================
#   1/5 — Data Ingestion
# =====================================================
# INFO - Ingested 7,043 rows
# =====================================================
#   2/5 — Feature Engineering
# =====================================================
# INFO - Features shape: (7043, 38)
# ...
# ✅ Pipeline complete in 245s
#    MLflow    → http://localhost:5000
#    API Docs  → http://localhost:8000/docs
#    Dashboard → http://localhost:8501
```

---

## PHASE 5 — Open All Services in Browser

After pipeline runs, open these in your browser:

| Service | URL | Login |
|---|---|---|
| **Airflow** | http://localhost:8080 | admin / admin |
| **MLflow** | http://localhost:5000 | no login |
| **API Docs** | http://localhost:8000/docs | no login |
| **Dashboard** | http://localhost:8501 | no login |
| **pgAdmin** (optional) | http://localhost:5050 | see below |

### Step 13 — Trigger Airflow DAG Manually
1. Open http://localhost:8080
2. Login: `admin` / `admin`
3. Find `churn_data_pipeline` in the DAG list
4. Click the **▶ Play** button on the right to trigger it manually
5. Click the DAG name to watch the task progress

---

## PHASE 6 — VS Code Connection

### Step 14 — Connect VS Code to Your Project
Since your project files are on Windows (`C:\Users\YourName\churn_prediction`),
just open the folder normally in VS Code:

```powershell
cd C:\Users\YourName\churn_prediction
code .
```

You edit files on Windows → Docker automatically picks up changes via **volume mounts**.
No restart needed for code changes in `src/`, `dags/`, `dashboard/`.

### Step 15 — Set Up Python Interpreter in VS Code
The containers run Python inside Docker, but for VS Code IntelliSense you need
local Python too:

```powershell
# Install Python 3.11 on Windows from https://www.python.org/downloads/
# Then in your project folder:
python -m venv venv_local
venv_local\Scripts\activate
pip install pandas numpy scikit-learn xgboost lightgbm mlflow fastapi pydantic loguru pyyaml evidently streamlit plotly
```

In VS Code:
- Press `Ctrl+Shift+P`
- Type: `Python: Select Interpreter`
- Choose: `.\venv_local\Scripts\python.exe`

Now you get full IntelliSense and autocomplete while Docker runs the actual code.

---

## Daily Workflow

### Starting the Stack
```powershell
cd C:\Users\YourName\churn_prediction

# Start all containers
docker compose up -d

# Wait 2 minutes, then open:
# http://localhost:8080  (Airflow)
# http://localhost:5000  (MLflow)
# http://localhost:8000/docs  (API)
# http://localhost:8501  (Dashboard)
```

### Stopping the Stack
```powershell
# Stop all containers (data is preserved in volumes)
docker compose down

# Stop AND delete all data (fresh start)
docker compose down -v
```

### Viewing Logs
```powershell
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f airflow-scheduler
docker compose logs -f mlflow
```

### Running Commands Inside a Container
```powershell
# Run pipeline
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py

# Open bash shell inside API container
docker exec -it churn_api bash

# Open bash inside Airflow
docker exec -it churn_airflow_web bash

# Test a specific Airflow task
docker exec -it churn_airflow_web airflow tasks test churn_data_pipeline ingest_raw_data 2024-01-01
```

### Making Code Changes
```powershell
# Edit files in VS Code normally on Windows
# Changes to src/, dags/, dashboard/ are immediate (volume mounted)

# If you change a Dockerfile or requirements, rebuild:
docker compose build api         # rebuild just API
docker compose build             # rebuild all
docker compose up -d             # restart with new images
```

---

## Troubleshooting

### Container Not Starting
```powershell
# Check which container failed
docker compose ps

# See detailed error logs
docker compose logs airflow-init
docker compose logs postgres
```

### Port Already in Use
```powershell
# Find what's using the port (e.g., 8080)
netstat -ano | findstr :8080

# Kill it (replace PID with the number from above)
taskkill /PID 12345 /F
```

### Airflow DAGs Not Showing
```powershell
# Check DAG files are mounted correctly
docker exec -it churn_airflow_web ls /opt/airflow/dags/

# Parse DAG manually to see errors
docker exec -it churn_airflow_web airflow dags list
docker exec -it churn_airflow_web python /opt/airflow/dags/churn_pipeline_dag.py
```

### Database Connection Error
```powershell
# Verify postgres is healthy
docker compose ps postgres

# Check postgres logs
docker compose logs postgres

# Connect to postgres directly
docker exec -it churn_postgres psql -U churn_user -d churn_db
```

### Out of Memory
```powershell
# Check container memory usage
docker stats

# If Docker Desktop is using too much RAM:
# Open Docker Desktop → Settings → Resources → Memory
# Set to at least 4GB (6GB recommended)
```

### Fresh Start (Nuclear Option)
```powershell
# Stop everything and delete ALL data
docker compose down -v

# Remove all images (forces full rebuild)
docker compose down --rmi all -v

# Start fresh
docker compose build
docker compose up -d
```

---

## Resource Requirements

| Minimum | Recommended |
|---|---|
| 8GB RAM | 16GB RAM |
| 4-core CPU | 8-core CPU |
| 10GB disk | 20GB disk |
| Windows 10 2004+ | Windows 11 |

In Docker Desktop → Settings → Resources:
- Memory: set to at least **6GB**
- CPUs: set to at least **4**

---

## What Each Container Does

```
churn_postgres          → Stores Airflow metadata, MLflow runs, prediction logs
churn_mlflow            → Tracks experiments, stores models in registry
churn_airflow_init      → Runs ONCE: creates DB tables + admin user, then exits
churn_airflow_web       → Airflow UI at :8080, also used to exec pipeline commands
churn_airflow_scheduler → Triggers DAGs on schedule (Mon 2am + Mon 4am UTC)
churn_api               → FastAPI serving predictions at :8000
churn_dashboard         → Streamlit monitoring dashboard at :8501
```

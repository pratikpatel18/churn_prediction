# 🚀 Complete Run Guide — From Unzip to Live Project
## Every command, every step, nothing skipped

---

# PRE-CHECKS (Do these before anything else)

## Check 1 — Your Windows Version
Open PowerShell (press Windows key, type "powershell", click it):
```powershell
winver
```
You need: **Windows 10 version 2004 or higher**, or Windows 11.
The version number shows in the popup window.

## Check 2 — Check if Python is installed
```powershell
python --version
```
If you see `Python 3.11.x` → good, skip to Check 3.
If you see "not recognized" → download Python 3.11 from https://www.python.org/downloads/
During install: ✅ check "Add Python to PATH"

## Check 3 — Check if Docker Desktop is installed
```powershell
docker --version
```
If you see `Docker version 26.x.x` → good, skip to STEP 1.
If you see "not recognized" → install Docker Desktop first:
  1. Go to https://www.docker.com/products/docker-desktop/
  2. Click "Download for Windows — AMD64"
  3. Run the installer
  4. During install: ✅ check "Use WSL 2 instead of Hyper-V"
  5. Click Install → Restart your PC when asked
  6. After restart: Open Docker Desktop from Start Menu
  7. Wait for green "Engine running" at bottom-left (takes ~2 mins)
  8. Go to Docker Desktop → Settings → Resources → Memory → set to 6144 MB → Apply & Restart

---

# STAGE 1 — UNZIP THE PROJECT

## Step 1 — Find Where You Downloaded the ZIP
The file is called: `churn_prediction_FINAL.zip`
It is probably in your Downloads folder.

## Step 2 — Unzip to Desktop
Open PowerShell and run this exact command:
```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\churn_prediction_FINAL.zip" -DestinationPath "$env:USERPROFILE\Desktop\" -Force
```

This creates a folder called `churn_prediction` on your Desktop.

## Step 3 — Verify the unzip worked
```powershell
dir "$env:USERPROFILE\Desktop\churn_prediction"
```
You should see folders like: configs, dags, dashboard, docker, src, scripts, data, etc.
If you see them → unzip worked ✅

---

# STAGE 2 — OPEN IN VS CODE

## Step 4 — Navigate to Project Folder
```powershell
cd "$env:USERPROFILE\Desktop\churn_prediction"
```

## Step 5 — Verify you are in the right place
```powershell
pwd
```
Output must show something like: `C:\Users\YourName\Desktop\churn_prediction`

## Step 6 — Open VS Code
```powershell
code .
```
VS Code opens with the full project in the left sidebar (Explorer panel).
You should see all folders: configs, dags, dashboard, docker, src, etc.

If "code" is not recognized:
  1. Open VS Code manually from Start Menu
  2. Click File → Open Folder
  3. Navigate to Desktop → churn_prediction → click "Select Folder"

---

# STAGE 3 — INSTALL VS CODE EXTENSIONS

## Step 7 — Install Extensions
Inside VS Code, press `Ctrl + Shift + X` (opens Extensions panel).
Search and install each of these one by one:

```
1. Search: "Python"        → Install "Python" by Microsoft
2. Search: "Pylance"       → Install "Pylance" by Microsoft
3. Search: "Docker"        → Install "Docker" by Microsoft
4. Search: "YAML"          → Install "YAML" by Red Hat
5. Search: "Jupyter"       → Install "Jupyter" by Microsoft
6. Search: "Thunder Client" → Install "Thunder Client" by Rangav
```

After installing, restart VS Code:
  Press `Ctrl + Shift + P` → type "Reload Window" → press Enter

---

# STAGE 4 — CREATE PYTHON ENVIRONMENT (for VS Code IntelliSense)

## Step 8 — Open VS Code Terminal
Inside VS Code: Press `Ctrl + ~` (backtick key, top-left of keyboard)
A terminal opens at the bottom of VS Code.

Make sure the terminal shows your project path:
```
C:\Users\YourName\Desktop\churn_prediction>
```
If it does not, run:
```powershell
cd "$env:USERPROFILE\Desktop\churn_prediction"
```

## Step 9 — Create Virtual Environment
```powershell
python -m venv venv_local
```
Wait ~10 seconds. A new folder called `venv_local` appears in your project.

## Step 10 — Activate the Virtual Environment
```powershell
venv_local\Scripts\activate
```
Your terminal prompt changes to show `(venv_local)` at the start:
```
(venv_local) C:\Users\YourName\Desktop\churn_prediction>
```

## Step 11 — Install Python Packages (for IntelliSense only)
```powershell
pip install --upgrade pip
```
```powershell
pip install pandas numpy scikit-learn xgboost lightgbm mlflow optuna fastapi pydantic uvicorn loguru pyyaml evidently streamlit plotly imbalanced-learn joblib great-expectations requests python-dotenv
```
This takes 3-5 minutes. You will see packages downloading.

## Step 12 — Set VS Code Python Interpreter
  1. Press `Ctrl + Shift + P`
  2. Type: `Python: Select Interpreter`
  3. Press Enter
  4. Choose the option that shows: `.\venv_local\Scripts\python.exe`
     (It should say "venv_local" in the path)

Now VS Code shows proper autocomplete and no red underlines on imports.

---

# STAGE 5 — ADD YOUR DATASET

## Step 13 — Copy Your CSV File
You need the Telco Customer Churn CSV file.
Dataset name: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
Download it from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Once downloaded, run this command in VS Code terminal:
```powershell
copy "$env:USERPROFILE\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv" "data\raw\churn_data.csv"
```

## Step 14 — Verify Dataset is in Place
```powershell
dir data\raw\
```
You must see: `churn_data.csv`
If you don't see it → the copy command failed. Check the source path is correct.

---

# STAGE 6 — CREATE .ENV FILE

## Step 15 — Copy the Environment Template
```powershell
copy .env.example .env
```

## Step 16 — Verify .env was created
```powershell
dir .env
```
You should see `.env` file listed.

## Step 17 — Open and Check .env File
In VS Code sidebar, click on `.env` to open it.
It should contain these exact values — do NOT change anything:
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
Save it: `Ctrl + S`

---

# STAGE 7 — BUILD DOCKER IMAGES

## Step 18 — Make Sure Docker Desktop is Running
Look at your Windows taskbar (bottom right, near clock).
You should see the Docker whale icon 🐳.
If not visible → open Docker Desktop from Start Menu and wait for "Engine running".

## Step 19 — Verify Docker is Working (in VS Code terminal)
```powershell
docker --version
```
```powershell
docker compose version
```
Both commands must return a version number. If either fails → Docker Desktop is not running.

## Step 20 — Build All Docker Images
⚠️ This is the LONGEST step. Takes 10-20 minutes first time.
Only needs to be done ONCE ever.

```powershell
docker compose build
```

You will see a lot of output scrolling. This is normal. Example:
```
[+] Building 2.3s (5/12) => [churn_api] Installing pip packages...
[+] Building 45.1s (8/12) => [churn_airflow] Installing airflow...
```

Wait until you see:
```
[+] Building XX.Xs (XX/XX) FINISHED
```
with no ERROR lines.

If you see an ERROR → check the Troubleshooting section at the bottom.

## Step 21 — Verify Images Were Built
```powershell
docker images
```
You should see these image names:
```
REPOSITORY                         TAG      SIZE
churn_prediction-airflow-webserver latest   ~2.5GB
churn_prediction-api               latest   ~1.2GB
churn_prediction-mlflow            latest   ~800MB
churn_prediction-streamlit         latest   ~900MB
postgres                           15-alpine ~240MB
```

---

# STAGE 8 — START ALL SERVICES

## Step 22 — Start the Entire Stack
```powershell
docker compose up -d
```

Output:
```
[+] Running 7/7
 ✔ Container churn_postgres           Started
 ✔ Container churn_mlflow             Started
 ✔ Container churn_airflow_init       Started
 ✔ Container churn_airflow_web        Started
 ✔ Container churn_airflow_scheduler  Started
 ✔ Container churn_api                Started
 ✔ Container churn_dashboard          Started
```

## Step 23 — Wait for Services to Start
Services need 2-4 minutes to fully start. Run this to watch progress:
```powershell
docker compose logs -f
```
Press `Ctrl + C` to stop watching logs (containers keep running).

## Step 24 — Check All Containers Are Healthy
```powershell
docker compose ps
```

Expected output (wait until all show these statuses):
```
NAME                      STATUS
churn_postgres            running (healthy)
churn_mlflow              running (healthy)
churn_airflow_init        exited (0)          ← NORMAL. Runs once then exits.
churn_airflow_web         running (healthy)
churn_airflow_scheduler   running
churn_api                 running (healthy)
churn_dashboard           running
```

⚠️ If something shows "starting" → wait 2 more minutes and run `docker compose ps` again.
⚠️ `airflow_init` showing `exited (0)` is NORMAL and CORRECT. It only runs once.

---

# STAGE 9 — RUN THE ML PIPELINE

## Step 25 — Run the Full Pipeline
This trains all 3 models and saves everything. Takes 3-8 minutes.

```powershell
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py
```

Watch the output. You will see:

```
=====================================================
  1/5 — Data Ingestion
=====================================================
INFO - Ingesting CSV from: data/raw/churn_data.csv
INFO - Loaded 7,043 rows x 21 columns
INFO - Saved → data/raw_parquet/churn_raw.parquet

=====================================================
  2/5 — Feature Engineering
=====================================================
INFO - Features shape: (7043, 38)
INFO - Features saved → data/features/churn_features.parquet

=====================================================
  3/5 — Reference Dataset Export
=====================================================
INFO - Reference: (2113, 38) → data/reference/reference_dataset.parquet

=====================================================
  4/5 — Model Training (MLflow + Optuna)
=====================================================
INFO - Churn rate: 26.54%
INFO - After SMOTE: churn rate 50.00%
INFO - Tuning: LOGISTIC_REGRESSION ... Best F1=0.5812
INFO - Tuning: XGBOOST ...           Best F1=0.6241
INFO - Tuning: LIGHTGBM ...          Best F1=0.6389
INFO - Selected: lightgbm
INFO - Registered 'churn_classifier' v1 → Staging

  MODEL RESULTS
  ─────────────────────────
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

=====================================================
  5/5 — Drift Monitoring
=====================================================
INFO - No drift alerts

✅ Pipeline complete!
   MLflow    → http://localhost:5000
   API Docs  → http://localhost:8000/docs
   Dashboard → http://localhost:8501
```

If you see this → pipeline worked perfectly ✅

---

# STAGE 10 — PROMOTE MODEL TO PRODUCTION

## Step 26 — Promote Best Model to Production
The pipeline saves the model to "Staging". Run this to move it to "Production":

```powershell
docker exec -it churn_airflow_web python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow:5000'); client = mlflow.tracking.MlflowClient(); versions = client.get_latest_versions('churn_classifier', stages=['Staging']); v = versions[0]; client.transition_model_version_stage(name='churn_classifier', version=v.version, stage='Production'); print(f'Model v{v.version} promoted to Production!')"
```

Output:
```
Model v1 promoted to Production!
```

---

# STAGE 11 — OPEN ALL SERVICES IN BROWSER

## Step 27 — Open All 4 Services
Open your browser (Chrome or Edge) and open these 4 tabs:

**Tab 1 — Airflow (Pipeline Scheduler)**
```
http://localhost:8080
Username: admin
Password: admin
```
You will see both DAGs listed: `churn_data_pipeline` and `churn_auto_retrain`

**Tab 2 — MLflow (Experiment Tracking)**
```
http://localhost:5000
```
Click "Experiments" → "churn_prediction_v1" → see all 3 model runs
Click "Models" → "churn_classifier" → see Production model

**Tab 3 — FastAPI (Prediction API)**
```
http://localhost:8000/docs
```
This is the interactive API documentation.
Click "POST /predict" → "Try it out" → fill in values → "Execute"

**Tab 4 — Streamlit (Monitoring Dashboard)**
```
http://localhost:8501
```
Live dashboard with drift monitoring, model metrics, experiment comparison

---

# STAGE 12 — TEST THE API

## Step 28 — Test Health Endpoint
Open a new terminal in VS Code (`Ctrl + ~`) and run:
```powershell
curl http://localhost:8000/health
```
Expected:
```json
{"status":"ok","model_loaded":true,"mlflow_uri":"http://mlflow:5000","timestamp":"..."}
```

## Step 29 — Test Prediction Endpoint
```powershell
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"gender\":\"Male\",\"SeniorCitizen\":0,\"Partner\":\"Yes\",\"Dependents\":\"No\",\"tenure\":6,\"PhoneService\":\"Yes\",\"MultipleLines\":\"No\",\"InternetService\":\"Fiber optic\",\"OnlineSecurity\":\"No\",\"OnlineBackup\":\"No\",\"DeviceProtection\":\"No\",\"TechSupport\":\"No\",\"StreamingTV\":\"No\",\"StreamingMovies\":\"No\",\"Contract\":\"Month-to-month\",\"PaperlessBilling\":\"Yes\",\"PaymentMethod\":\"Electronic check\",\"MonthlyCharges\":70.35,\"TotalCharges\":422.10}"
```
Expected response:
```json
{
  "request_id": "some-uuid",
  "churn_probability": 0.8732,
  "churn_prediction": 1,
  "risk_label": "Critical",
  "timestamp": "2024-..."
}
```

---

# STAGE 13 — TRIGGER AIRFLOW DAG

## Step 30 — Manually Trigger the Data Pipeline DAG
```
1. Open http://localhost:8080
2. Login: admin / admin
3. Find "churn_data_pipeline" in the list
4. Click the grey toggle on the LEFT to turn it BLUE (unpause it)
5. Click the ▶ play button on the RIGHT side of the row
6. Click "Trigger DAG" in the popup
7. Click the DAG name "churn_data_pipeline" to open it
8. Click "Graph" tab to watch tasks run
9. Green box = task passed, Red box = task failed
```

---

# DAILY WORKFLOW COMMANDS

## Every Morning — Start the Stack
Open PowerShell or VS Code Terminal:
```powershell
cd "$env:USERPROFILE\Desktop\churn_prediction"
docker compose up -d
```
Wait 2-3 minutes, then open browser tabs.

## Every Evening — Stop the Stack
```powershell
docker compose down
```
Data is saved. Nothing is lost.

## Check Container Status Any Time
```powershell
docker compose ps
```

## View Logs for Debugging
```powershell
# All containers
docker compose logs -f

# Specific container
docker compose logs -f api
docker compose logs -f airflow-webserver
docker compose logs -f mlflow
docker compose logs -f postgres
```

## Restart One Container
```powershell
docker compose restart api
docker compose restart airflow-scheduler
```

## Run Pipeline Again (after new data)
```powershell
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py
```

## Run Monitoring Report
```powershell
docker exec -it churn_airflow_web python scripts/run_monitoring.py
```

## Open Shell Inside Any Container
```powershell
# Inside Airflow
docker exec -it churn_airflow_web bash

# Inside API
docker exec -it churn_api bash

# Inside database
docker exec -it churn_postgres psql -U churn_user -d churn_db
```
Type `exit` to leave the container shell.

---

# TROUBLESHOOTING — Most Common Errors

## ❌ Error: "docker: command not found" or "not recognized"
Docker Desktop is not running.
→ Open Docker Desktop from Start Menu
→ Wait for green "Engine running"
→ Try again

## ❌ Error: "port is already allocated" or "bind: address already in use"
Another program is using that port.
```powershell
# Find what is using port 8080 (replace 8080 with the port shown in error)
netstat -ano | findstr :8080

# Note the PID number in the last column, then kill it:
taskkill /PID 1234 /F

# Then start again:
docker compose up -d
```

## ❌ Error: "churn_airflow_init exited with code 1" (not code 0)
The database was not ready in time.
```powershell
# Stop everything
docker compose down

# Start fresh
docker compose up -d

# Watch init logs specifically
docker compose logs -f airflow-init
```

## ❌ Error: "No module named 'src'" when running pipeline
```powershell
# Check PYTHONPATH inside container
docker exec -it churn_airflow_web env | findstr PYTHONPATH

# Should show: PYTHONPATH=/opt/airflow
# If not, rebuild:
docker compose build
docker compose up -d
```

## ❌ Error: "churn_data.csv not found" when running pipeline
Your dataset is missing.
```powershell
# Check if file exists
dir data\raw\

# If empty, copy your CSV:
copy "C:\Users\%USERNAME%\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv" "data\raw\churn_data.csv"

# Verify
dir data\raw\
# Must show churn_data.csv
```

## ❌ Error: "Model not loaded" from API (503 error)
Pipeline hasn't run yet OR model not promoted to Production.
```powershell
# Step 1: Run pipeline first if not done
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py

# Step 2: Promote model
docker exec -it churn_airflow_web python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow:5000'); client = mlflow.tracking.MlflowClient(); versions = client.get_latest_versions('churn_classifier', stages=['Staging']); v = versions[0]; client.transition_model_version_stage(name='churn_classifier', version=v.version, stage='Production'); print('Done!')"

# Step 3: Restart API to load new model
docker compose restart api

# Step 4: Test again after 30 seconds
curl http://localhost:8000/health
```

## ❌ Error: "DAGs not showing" in Airflow UI
```powershell
# Check DAG files are inside container
docker exec -it churn_airflow_web ls /opt/airflow/dags/

# Should show:
# churn_pipeline_dag.py
# churn_retrain_dag.py

# If missing, check volume mount is correct in docker-compose.yml
# Then restart scheduler:
docker compose restart airflow-scheduler

# Wait 1 minute, refresh Airflow browser tab
```

## ❌ Error: Docker build fails with "network timeout"
Your internet was slow during build.
```powershell
# Just retry (it resumes from cache):
docker compose build

# If still failing, build with no cache:
docker compose build --no-cache
```

## ❌ Error: "Out of memory" or containers keep restarting
Docker needs more RAM.
→ Open Docker Desktop
→ Click Settings (gear icon top right)
→ Click Resources
→ Memory slider → drag to 6144 MB (6 GB) minimum
→ Click "Apply & Restart"
→ After Docker restarts, run `docker compose up -d` again

## ❌ Streamlit dashboard shows no data / all N/A
Run the pipeline first. Dashboard reads from files the pipeline creates.
```powershell
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py
```
Then refresh http://localhost:8501

## ❌ Airflow shows "dag_id not found" when testing tasks
Airflow hasn't scanned the DAG file yet.
```powershell
# Force rescan
docker exec -it churn_airflow_web airflow dags reserialize

# List all DAGs to confirm they appear
docker exec -it churn_airflow_web airflow dags list
```

---

# NUCLEAR OPTION — Complete Fresh Start

If nothing works and you want to start 100% clean:
```powershell
# WARNING: This deletes ALL data including trained models and MLflow history

# Stop and remove everything including volumes
docker compose down -v

# Remove all built images
docker compose down --rmi all -v

# Rebuild from scratch
docker compose build

# Start again
docker compose up -d

# Wait 4 minutes, then run pipeline
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py
```

---

# COMPLETE COMMAND SUMMARY (All Steps in Order)

Copy and run these one at a time in VS Code terminal:

```powershell
# STAGE 1 — Unzip
Expand-Archive -Path "$env:USERPROFILE\Downloads\churn_prediction_FINAL.zip" -DestinationPath "$env:USERPROFILE\Desktop\" -Force

# STAGE 2 — Open in VS Code
cd "$env:USERPROFILE\Desktop\churn_prediction"
code .

# STAGE 3 — Python environment (in VS Code terminal)
python -m venv venv_local
venv_local\Scripts\activate
pip install --upgrade pip
pip install pandas numpy scikit-learn xgboost lightgbm mlflow optuna fastapi pydantic uvicorn loguru pyyaml evidently streamlit plotly imbalanced-learn joblib requests python-dotenv

# STAGE 4 — Add dataset (replace path if needed)
copy "$env:USERPROFILE\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv" "data\raw\churn_data.csv"

# STAGE 5 — Create .env
copy .env.example .env

# STAGE 6 — Build Docker (ONCE, takes 15 mins)
docker compose build

# STAGE 7 — Start all services
docker compose up -d

# Wait 3 minutes, then:

# STAGE 8 — Check everything is running
docker compose ps

# STAGE 9 — Run ML pipeline
docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py

# STAGE 10 — Promote model to Production
docker exec -it churn_airflow_web python -c "import mlflow; mlflow.set_tracking_uri('http://mlflow:5000'); client = mlflow.tracking.MlflowClient(); versions = client.get_latest_versions('churn_classifier', stages=['Staging']); v = versions[0]; client.transition_model_version_stage(name='churn_classifier', version=v.version, stage='Production'); print(f'Model v{v.version} promoted!')"

# STAGE 11 — Test API
curl http://localhost:8000/health

# Open browser:
# http://localhost:8080  (Airflow - admin/admin)
# http://localhost:5000  (MLflow)
# http://localhost:8000/docs  (API)
# http://localhost:8501  (Streamlit Dashboard)
```

---

## You are done! 🎉

All 7 containers running. All 4 browser tabs open. Model trained and live.

@echo off
REM ─────────────────────────────────────────────────────────────────
REM  start_windows.bat
REM  Double-click this file to start the entire Churn Prediction stack
REM  Requires: Docker Desktop running
REM ─────────────────────────────────────────────────────────────────

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║   Customer Churn Prediction - Starting Stack...      ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

REM Check Docker is running
docker info > nul 2>&1
if %errorlevel% neq 0 (
    echo  ❌ Docker Desktop is not running!
    echo  Please start Docker Desktop and wait for it to show "Engine running"
    echo  Then double-click this file again.
    pause
    exit /b 1
)

echo  ✅ Docker Desktop is running
echo.

REM Check .env file exists
if not exist .env (
    echo  ⚠️  .env file not found - creating from template...
    copy .env.example .env
    echo  ✅ .env created - using default values
    echo.
)

REM Check dataset exists
if not exist data\raw\churn_data.csv (
    echo  ❌ Dataset not found at data\raw\churn_data.csv
    echo.
    echo  Please copy your Telco CSV file to:
    echo  %CD%\data\raw\churn_data.csv
    echo.
    pause
    exit /b 1
)

echo  ✅ Dataset found
echo.
echo  🐳 Starting all Docker containers...
echo.

REM Start containers
docker compose up -d

if %errorlevel% neq 0 (
    echo.
    echo  ❌ Failed to start containers.
    echo  Run this command to see errors:
    echo  docker compose logs
    pause
    exit /b 1
)

echo.
echo  ⏳ Waiting for services to become healthy (60 seconds)...
timeout /t 60 /nobreak > nul

echo.
echo  ✅ Stack started! Opening browser tabs...
echo.

REM Open all services in browser
start "" "http://localhost:8080"
timeout /t 2 /nobreak > nul
start "" "http://localhost:5000"
timeout /t 2 /nobreak > nul
start "" "http://localhost:8000/docs"
timeout /t 2 /nobreak > nul
start "" "http://localhost:8501"

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║   All services started!                              ║
echo  ║                                                      ║
echo  ║   Airflow    → http://localhost:8080                 ║
echo  ║               Login: admin / admin                   ║
echo  ║                                                      ║
echo  ║   MLflow     → http://localhost:5000                 ║
echo  ║                                                      ║
echo  ║   API Docs   → http://localhost:8000/docs            ║
echo  ║                                                      ║
echo  ║   Dashboard  → http://localhost:8501                 ║
echo  ╚══════════════════════════════════════════════════════╝
echo.
echo  To run the ML pipeline, open a new terminal and run:
echo  docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py
echo.
echo  To stop everything:  docker compose down
echo.
pause

@echo off
REM ─────────────────────────────────────────────────────────────────
REM  run_pipeline.bat
REM  Double-click to run the full ML pipeline inside Docker
REM  Must run AFTER start_windows.bat
REM ─────────────────────────────────────────────────────────────────

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║   Running Full ML Pipeline...                        ║
echo  ║   This takes 3-5 minutes. Please wait.               ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

REM Check Airflow container is running
docker ps --filter "name=churn_airflow_web" --filter "status=running" | findstr "churn_airflow_web" > nul
if %errorlevel% neq 0 (
    echo  ❌ Airflow container is not running!
    echo  Please run start_windows.bat first.
    pause
    exit /b 1
)

echo  Running pipeline inside Airflow container...
echo  Watch the output below:
echo.
echo  ─────────────────────────────────────────────────────
echo.

docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py

echo.
echo  ─────────────────────────────────────────────────────
echo.
echo  ✅ Pipeline complete!
echo.
echo  Next step - promote model to Production:
echo  Open http://localhost:5000 → Models → churn_classifier
echo  Click Version 1 → Transition to Production
echo.
pause

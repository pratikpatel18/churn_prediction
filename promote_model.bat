@echo off
REM ─────────────────────────────────────────────────────────────────
REM  promote_model.bat
REM  Promotes the best Staging model to Production in MLflow
REM  Run AFTER run_pipeline.bat completes
REM ─────────────────────────────────────────────────────────────────

echo.
echo  Promoting best model to Production in MLflow...
echo.

docker exec -it churn_airflow_web python -c ^
"import mlflow; ^
mlflow.set_tracking_uri('http://mlflow:5000'); ^
client = mlflow.tracking.MlflowClient(); ^
versions = client.get_latest_versions('churn_classifier', stages=['Staging']); ^
v = versions[0] if versions else None; ^
client.transition_model_version_stage(name='churn_classifier', version=v.version, stage='Production') if v else print('No staging model found. Run pipeline first.'); ^
print(f'Model v{v.version} promoted to Production!') if v else None"

echo.
echo  ✅ Done! Your API now serves the Production model.
echo  Test it at: http://localhost:8000/docs
echo.
pause

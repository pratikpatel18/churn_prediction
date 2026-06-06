import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='churn_classifier'")
print(f'Total versions: {len(versions)}')
for v in versions:
    print(f'v{v.version} - {v.current_stage}')
runs = mlflow.search_runs(experiment_names=['churn_prediction_v1'])
print(f'Total runs: {len(runs)}')

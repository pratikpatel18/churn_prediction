import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='churn_classifier'")
prod = [x for x in versions if x.current_stage == 'Production'][0]
model = mlflow.pyfunc.load_model(f'runs:/{prod.run_id}/model')
schema = model.metadata.get_input_schema()
cols = [c.name for c in schema.inputs]
print('COUNT:', len(cols))
for c in cols:
    print(c)

import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions("name='churn_classifier'")
latest = sorted(versions, key=lambda v: int(v.version))[-1]
run_id = latest.run_id
print("Run ID:", run_id)

model = mlflow.pyfunc.load_model("runs:/" + run_id + "/model")
schema = model.metadata.get_input_schema()
print("Input schema:")
for col in schema.inputs:
    print("  " + col.name + ": " + str(col.type))
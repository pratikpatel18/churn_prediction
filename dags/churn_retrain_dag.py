from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, str(Path(__file__).parents[1]))

import mlflow
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

ACCURACY_THRESHOLD = CONFIG["monitoring"]["accuracy_threshold"]
F1_THRESHOLD = CONFIG["monitoring"]["f1_threshold"]


# ─────────────────────────────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────────────────────────────


def task_evaluate_current(**ctx):
    """
    Load the Production model and score it on fresh data.
    Pushes current metrics to XCom so the branch task can decide.
    """
    mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])

    # Load latest production model
    model_name = CONFIG["mlflow"]["model_name"]
    model_uri = f"models:/{model_name}/Production"

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.error(f"Could not load Production model: {e}")
        # No production model → force retraining
        ctx["ti"].xcom_push(key="current_f1", value=0.0)
        ctx["ti"].xcom_push(key="current_accuracy", value=0.0)
        return

    # Score on latest feature set
    feat_path = CONFIG["paths"]["features_data"]
    df = pd.read_parquet(feat_path)
    X = df.drop(columns=[CONFIG["data"]["target_column"]])
    y = df[CONFIG["data"]["target_column"]].astype(int)

    # Use last 20% as fresh "production" slice
    split = int(len(df) * 0.8)
    X_new, y_new = X.iloc[split:], y.iloc[split:]

    y_pred = model.predict(X_new)
    if hasattr(y_pred, "ndim") and y_pred.ndim == 2:
        y_pred = (y_pred[:, 1] >= 0.5).astype(int)
    else:
        y_pred = (y_pred >= 0.5).astype(int)

    current_f1 = round(f1_score(y_new, y_pred), 4)
    current_acc = round(accuracy_score(y_new, y_pred), 4)

    logger.info(f"Current model — F1={current_f1}  Accuracy={current_acc}")

    ctx["ti"].xcom_push(key="current_f1", value=current_f1)
    ctx["ti"].xcom_push(key="current_accuracy", value=current_acc)

    # Log to MLflow for tracking
    with mlflow.start_run(run_name="weekly_perf_check"):
        mlflow.log_metric("weekly_f1", current_f1)
        mlflow.log_metric("weekly_accuracy", current_acc)
        mlflow.log_metric("f1_threshold", F1_THRESHOLD)


def task_branch(**ctx):
    """Branch: retrain if F1 dropped below threshold, else skip."""
    current_f1 = ctx["ti"].xcom_pull(key="current_f1")

    if current_f1 < F1_THRESHOLD:
        logger.warning(f"F1={current_f1} < threshold={F1_THRESHOLD} → triggering retraining")
        return "retrain_model"
    else:
        logger.info(f"F1={current_f1} ≥ threshold={F1_THRESHOLD} → model OK, skipping retrain")
        return "skip_retrain"


def task_retrain(**ctx):
    """Run full Optuna + MLflow training on the latest features."""
    from src.models.train import ModelTrainer

    feat_path = CONFIG["paths"]["features_data"]
    trainer = ModelTrainer(CONFIG)
    results = trainer.run(feat_path)

    best_name = max(results, key=lambda k: results[k]["metrics"]["f1"])
    best_metrics = results[best_name]["metrics"]
    best_run_id = results[best_name]["run_id"]

    logger.success(f"Retrain complete. Best: {best_name} F1={best_metrics['f1']}")

    ctx["ti"].xcom_push(key="retrain_f1", value=best_metrics["f1"])
    ctx["ti"].xcom_push(key="retrain_run_id", value=best_run_id)
    ctx["ti"].xcom_push(key="retrain_model", value=best_name)


def task_promote(**ctx):
    """
    Promote new Staging model to Production IF it's better than current.
    Archive the old Production model.
    """
    retrain_f1 = ctx["ti"].xcom_pull(key="retrain_f1")
    current_f1 = ctx["ti"].xcom_pull(key="current_f1")

    if retrain_f1 <= current_f1:
        logger.info(f"New model F1={retrain_f1} ≤ current F1={current_f1}. Not promoting.")
        return

    client = mlflow.tracking.MlflowClient()
    model_name = CONFIG["mlflow"]["model_name"]

    # Archive existing Production
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for v in prod_versions:
        client.transition_model_version_stage(name=model_name, version=v.version, stage="Archived")
        logger.info(f"Archived v{v.version}")

    # Promote Staging → Production
    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    if staging_versions:
        new_v = staging_versions[-1]
        client.transition_model_version_stage(
            name=model_name, version=new_v.version, stage="Production"
        )
        logger.success(f"Promoted v{new_v.version} → Production (F1 {current_f1} → {retrain_f1})")

        # Add descriptive alias
        client.update_model_version(
            name=model_name,
            version=new_v.version,
            description=(
                f"Auto-promoted on {datetime.utcnow().date()} | "
                f"F1={retrain_f1} | replaced F1={current_f1}"
            ),
        )


def task_notify_retrain(**ctx):
    retrain_f1 = ctx["ti"].xcom_pull(key="retrain_f1")
    current_f1 = ctx["ti"].xcom_pull(key="current_f1") or "N/A"
    model_name = ctx["ti"].xcom_pull(key="retrain_model")
    run_date = ctx["ds"]

    msg = (
        f"🔄 Churn Model RETRAINED & PROMOTED [{run_date}]\n"
        f"  Old F1   : {current_f1}\n"
        f"  New F1   : {retrain_f1}\n"
        f"  Model    : {model_name}\n"
        f"  Stage    : Production"
    )
    logger.success(msg)
    # TODO: POST to Slack webhook


# ─────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="churn_auto_retrain",
    default_args=default_args,
    description="Automated weekly model evaluation and conditional retraining",
    schedule_interval=CONFIG["airflow"]["retraining_schedule"],  # "0 4 * * 1"
    catchup=False,
    tags=["churn", "retraining", "mlops"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    evaluate = PythonOperator(
        task_id="evaluate_current_model",
        python_callable=task_evaluate_current,
        provide_context=True,
    )

    branch = BranchPythonOperator(
        task_id="check_performance",
        python_callable=task_branch,
        provide_context=True,
    )

    skip = EmptyOperator(task_id="skip_retrain")

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=task_retrain,
        provide_context=True,
    )

    promote = PythonOperator(
        task_id="promote_new_model",
        python_callable=task_promote,
        provide_context=True,
    )

    notify = PythonOperator(
        task_id="notify_retrain",
        python_callable=task_notify_retrain,
        provide_context=True,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    # ── DAG wiring ────────────────────────────────────────────────
    start >> evaluate >> branch
    branch >> skip >> end
    branch >> retrain >> promote >> notify >> end

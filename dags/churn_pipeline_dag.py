from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

# Ensure project root is on PYTHONPATH inside Airflow workers
sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import yaml
from loguru import logger

# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# ─────────────────────────────────────────────────────────────────────
# DAG default args
# ─────────────────────────────────────────────────────────────────────

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": True,
    "email": ["mlops@yourcompany.com"],
    "retries": CONFIG["airflow"]["retries"],
    "retry_delay": timedelta(minutes=CONFIG["airflow"]["retry_delay_minutes"]),
}


# ─────────────────────────────────────────────────────────────────────
# Task functions
# ─────────────────────────────────────────────────────────────────────


def task_ingest_raw(**ctx):
    """
    Pull latest customer data from source and save as raw Parquet.
    In production: replace with DB query or S3 fetch.
    """
    from src.data.ingest import DataIngester

    ingester = DataIngester(CONFIG)

    raw_path = CONFIG["paths"]["raw_data"]
    df = ingester.ingest_csv(raw_path)

    out_path = CONFIG["paths"]["processed_data"].replace("processed", "raw_parquet")
    ingester.save_processed(df, out_path)

    ctx["ti"].xcom_push(key="raw_parquet_path", value=out_path)
    ctx["ti"].xcom_push(key="n_rows", value=len(df))
    logger.info(f"Ingested {len(df)} rows → {out_path}")


def task_validate_raw(**ctx):
    """Run Great Expectations against raw ingested data."""
    from src.data.validate import DataValidator

    raw_path = ctx["ti"].xcom_pull(key="raw_parquet_path")
    df = pd.read_parquet(raw_path)

    validator = DataValidator()
    result = validator.validate_raw(df)

    if not result["success"]:
        n_failed = result["n_failed"]
        raise ValueError(
            f"Raw data validation FAILED — {n_failed} expectations violated. "
            f"Pipeline halted to prevent bad data from propagating."
        )

    logger.success("Raw data validation passed ✓")
    ctx["ti"].xcom_push(key="raw_validation_passed", value=True)


def task_engineer_features(**ctx):
    """Build feature set from cleaned data."""
    from src.features.engineer import build_features

    raw_path = ctx["ti"].xcom_pull(key="raw_parquet_path")
    feat_path = CONFIG["paths"]["features_data"]

    df_features = build_features(
        input_path=raw_path,
        output_path=feat_path,
        config=CONFIG,
        artifact_dir=CONFIG["paths"]["model_artifacts"],
    )

    ctx["ti"].xcom_push(key="features_path", value=feat_path)
    ctx["ti"].xcom_push(key="n_features", value=df_features.shape[1])
    logger.info(f"Features: {df_features.shape}")


def task_validate_features(**ctx):
    """Validate engineered features before training."""
    from src.data.validate import DataValidator

    feat_path = ctx["ti"].xcom_pull(key="features_path")
    df = pd.read_parquet(feat_path)

    validator = DataValidator()
    result = validator.validate_features(df)

    if not result["success"]:
        raise ValueError(
            f"Feature validation FAILED — {result['n_failed']} checks violated."
        )

    logger.success("Feature validation passed ✓")


def task_dvc_version(**ctx):
    """
    Commit the new dataset to DVC and push to remote.
    This creates a versioned snapshot for reproducibility.
    """
    import subprocess

    feat_path = ctx["ti"].xcom_pull(key="features_path")
    run_date = ctx["ds"]  # Airflow execution date (YYYY-MM-DD)

    commands = [
        f"dvc add {feat_path}",
        f"git add {feat_path}.dvc .gitignore",
        f'git commit -m "data: weekly feature update {run_date} [airflow]"',
        "dvc push",
    ]

    for cmd in commands:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"DVC cmd failed (non-fatal): {result.stderr}")
        else:
            logger.info(f"✓ {cmd}")


def task_export_reference(**ctx):
    """
    Save a 30% random sample as the reference dataset for Evidently drift detection.
    This snapshot represents 'what the model trained on'.
    """
    feat_path = ctx["ti"].xcom_pull(key="features_path")
    ref_path = CONFIG["paths"]["reference_data"]

    df = pd.read_parquet(feat_path)
    ref_df = df.sample(frac=0.3, random_state=42)

    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    ref_df.to_parquet(ref_path, index=False)

    logger.success(f"Reference dataset saved: {ref_df.shape} → {ref_path}")
    ctx["ti"].xcom_push(key="reference_path", value=ref_path)


def task_notify_success(**ctx):
    """Send Slack/email notification on pipeline success."""
    n_rows = ctx["ti"].xcom_pull(key="n_rows")
    n_features = ctx["ti"].xcom_pull(key="n_features")
    run_date = ctx["ds"]

    msg = (
        f"✅ Churn Pipeline SUCCESS [{run_date}]\n"
        f"  Rows ingested  : {n_rows:,}\n"
        f"  Features built : {n_features}\n"
        f"  Run by         : Airflow scheduler"
    )
    logger.success(msg)
    # TODO: Replace with actual Slack webhook call
    # requests.post(SLACK_WEBHOOK_URL, json={"text": msg})


# ─────────────────────────────────────────────────────────────────────
# DAG definition
# ─────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="churn_data_pipeline",
    default_args=default_args,
    description="Weekly churn data pipeline: ingest → validate → feature engineering → DVC",
    schedule_interval=CONFIG["airflow"]["pipeline_schedule"],  # "0 2 * * 1"
    catchup=False,
    tags=["churn", "data-pipeline", "mlops"],
    doc_md=__doc__,
) as dag:

    start = EmptyOperator(task_id="start")

    ingest = PythonOperator(
        task_id="ingest_raw_data",
        python_callable=task_ingest_raw,
        provide_context=True,
    )

    validate_raw = PythonOperator(
        task_id="validate_raw_data",
        python_callable=task_validate_raw,
        provide_context=True,
    )

    engineer = PythonOperator(
        task_id="engineer_features",
        python_callable=task_engineer_features,
        provide_context=True,
    )

    validate_features = PythonOperator(
        task_id="validate_features",
        python_callable=task_validate_features,
        provide_context=True,
    )

    dvc_version = PythonOperator(
        task_id="dvc_version_data",
        python_callable=task_dvc_version,
        provide_context=True,
    )

    export_reference = PythonOperator(
        task_id="export_reference_dataset",
        python_callable=task_export_reference,
        provide_context=True,
    )

    notify = PythonOperator(
        task_id="notify_success",
        python_callable=task_notify_success,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    end = EmptyOperator(task_id="end")

    # ── Dependency chain ──────────────────────────────────────────
    (
        start
        >> ingest
        >> validate_raw
        >> engineer
        >> validate_features
        >> dvc_version
        >> export_reference
        >> notify
        >> end
    )

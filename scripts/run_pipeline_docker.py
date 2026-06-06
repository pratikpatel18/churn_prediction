"""
scripts/run_pipeline_docker.py
───────────────────────────────
Runs the full ML pipeline inside the Airflow Docker container.
Called manually for first-time setup or triggered by Airflow DAG.

Usage (from your Windows terminal after docker compose up):
    docker exec -it churn_airflow_web python scripts/run_pipeline_docker.py
"""

import os
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

# Project root is /opt/airflow inside the container
sys.path.insert(0, "/opt/airflow")


def banner(msg: str):
    logger.info(f"\n{'='*55}\n  {msg}\n{'='*55}")


def main():
    config_path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    start = time.time()

    # ── Step 1: Ingest ────────────────────────────────────────────
    banner("1/5 — Data Ingestion")
    from src.data.ingest import DataIngester

    data_path = config["paths"]["raw_data"]
    if not Path(data_path).exists():
        logger.error(
            f"Dataset not found at: {data_path}\n"
            "Make sure you copied your CSV to data/raw/churn_data.csv"
        )
        sys.exit(1)

    ingester = DataIngester(config)
    df_raw   = ingester.ingest_csv(data_path)

    raw_parquet = "data/raw_parquet/churn_raw.parquet"
    Path(raw_parquet).parent.mkdir(parents=True, exist_ok=True)
    ingester.save_processed(df_raw, raw_parquet)
    logger.info(f"Ingested {len(df_raw):,} rows")

    # ── Step 2: Feature Engineering ───────────────────────────────
    banner("2/5 — Feature Engineering")
    from src.features.engineer import build_features

    feat_path = config["paths"]["features_data"]
    df_features = build_features(
        input_path   = raw_parquet,
        output_path  = feat_path,
        config       = config,
        artifact_dir = config["paths"]["model_artifacts"],
    )
    logger.info(f"Features shape: {df_features.shape}")

    # ── Step 3: Export Reference Dataset ──────────────────────────
    banner("3/5 — Reference Dataset Export")
    import pandas as pd

    ref_path = config["paths"]["reference_data"]
    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    ref_df = df_features.sample(frac=0.3, random_state=42)
    ref_df.to_parquet(ref_path, index=False)
    logger.success(f"Reference dataset: {ref_df.shape} → {ref_path}")

    # ── Step 4: Model Training ────────────────────────────────────
    banner("4/5 — Model Training (MLflow + Optuna)")
    from src.models.train import ModelTrainer

    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
    os.environ["MLFLOW_ARTIFACT_URI"] = "http://mlflow:5000"
    import mlflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    trainer = ModelTrainer(config)
    results = trainer.run(feat_path)

    print("\n" + "─"*50)
    print("  MODEL RESULTS")
    print("─"*50)
    for name, data in results.items():
        m = data["metrics"]
        print(f"\n  {name.upper()}")
        for k, v in m.items():
            print(f"    {k:12s}: {v:.4f}")
    print("─"*50)

    # ── Step 5: Monitoring ────────────────────────────────────────
    banner("5/5 — Drift Monitoring")
    from src.monitoring.drift_monitor import DriftMonitor

    reference_df = pd.read_parquet(ref_path)
    current_df   = df_features

    monitor = DriftMonitor(config)
    report  = monitor.run_full_report(reference_df, current_df)
    monitor.save_report(report, "artifacts/reports/latest_drift_report.json")

    if report.get("alerts"):
        for alert in report["alerts"]:
            logger.warning(f"[{alert['level']}] {alert['msg']}")
    else:
        logger.success("No drift alerts")

    elapsed = time.time() - start
    logger.success(
        f"\n✅ Pipeline complete in {elapsed:.0f}s\n"
        f"   MLflow    → http://localhost:5000\n"
        f"   API Docs  → http://localhost:8000/docs\n"
        f"   Dashboard → http://localhost:8501\n"
    )


if __name__ == "__main__":
    main()

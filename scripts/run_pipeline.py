
import argparse
import sys
import time
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parents[1]))


def banner(step: str):
    logger.info(f"\n{'='*60}\n  STEP: {step}\n{'='*60}")


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    start_total = time.time()

    # ── Step 1: Ingest ────────────────────────────────────────────
    banner("1 / 6  —  Data Ingestion")
    from src.data.ingest import DataIngester

    ingester  = DataIngester(config)
    data_path = args.data_path or config["paths"]["raw_data"]
    df_raw    = ingester.ingest_csv(data_path)

    raw_parquet = "data/raw_parquet/churn_raw.parquet"
    ingester.save_processed(df_raw, raw_parquet)

    # ── Step 2: Validate Raw ──────────────────────────────────────
    if not args.skip_validation:
        banner("2 / 6  —  Raw Data Validation (Great Expectations)")
        from src.data.validate import DataValidator

        validator = DataValidator()
        try:
            result = validator.validate_raw(df_raw)
            if not result["success"]:
                logger.error(f"Raw validation failed! {result['n_failed']} checks violated.")
                if args.strict:
                    sys.exit(1)
        except Exception as e:
            logger.warning(f"Validation skipped (GE not configured): {e}")
    else:
        logger.info("Validation skipped (--skip-validation flag)")

    # ── Step 3: Feature Engineering ───────────────────────────────
    banner("3 / 6  —  Feature Engineering")
    from src.features.engineer import build_features

    df_features = build_features(
        input_path   = raw_parquet,
        output_path  = config["paths"]["features_data"],
        config       = config,
        artifact_dir = config["paths"]["model_artifacts"],
    )
    logger.info(f"Features shape: {df_features.shape}")

    # ── Step 4: Export Reference Dataset ──────────────────────────
    banner("4 / 6  —  Export Reference Dataset")
    import pandas as pd

    ref_df = df_features.sample(frac=0.3, random_state=42)
    ref_path = config["paths"]["reference_data"]
    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    ref_df.to_parquet(ref_path, index=False)
    logger.success(f"Reference dataset: {ref_df.shape} → {ref_path}")

    # ── Step 5: Model Training ────────────────────────────────────
    if not args.skip_training:
        banner("5 / 6  —  Model Training (MLflow + Optuna)")
        from src.models.train import ModelTrainer

        trainer = ModelTrainer(config)
        results = trainer.run(config["paths"]["features_data"])

        print("\n" + "─" * 50)
        print("  MODEL COMPARISON RESULTS")
        print("─" * 50)
        for model_name, data in results.items():
            m = data["metrics"]
            print(f"\n  {model_name.upper()}")
            for metric, val in m.items():
                print(f"    {metric:12s}: {val:.4f}")
        print("─" * 50)
    else:
        logger.info("Training skipped (--skip-training flag)")

    # ── Step 6: Run Monitoring Report ─────────────────────────────
    if not args.skip_monitoring:
        banner("6 / 6  —  Drift Monitoring (Evidently)")
        from src.monitoring.drift_monitor import DriftMonitor
        import json

        reference_df = pd.read_parquet(config["paths"]["reference_data"])
        current_df   = df_features  # In production: load fresh batch

        monitor = DriftMonitor(config)
        report  = monitor.run_full_report(reference_df, current_df)
        monitor.save_report(report, "artifacts/reports/latest_drift_report.json")

        if report["alerts"]:
            for alert in report["alerts"]:
                logger.warning(f"ALERT [{alert['level']}]: {alert['msg']}")
        else:
            logger.success("No drift alerts")
    else:
        logger.info("Monitoring skipped (--skip-monitoring flag)")

    # ── Done ───────────────────────────────────────────────────────
    elapsed = time.time() - start_total
    logger.success(
        f"\n✅ Pipeline complete in {elapsed:.1f}s\n"
        f"   MLflow UI  : http://localhost:5000\n"
        f"   API        : uvicorn src.api.main:app --reload\n"
        f"   Dashboard  : streamlit run dashboard/streamlit_app.py\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full churn pipeline")
    parser.add_argument("--config",           default="configs/config.yaml")
    parser.add_argument("--data-path",        default=None)
    parser.add_argument("--skip-training",    action="store_true")
    parser.add_argument("--skip-validation",  action="store_true")
    parser.add_argument("--skip-monitoring",  action="store_true")
    parser.add_argument("--strict",           action="store_true",
                        help="Exit on validation failure")
    args = parser.parse_args()
    main(args)

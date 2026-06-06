"""
scripts/run_monitoring.py
──────────────────────────
Standalone drift monitoring script.
Can be run manually or via Airflow / cron.

Usage:
    python scripts/run_monitoring.py
    python scripts/run_monitoring.py --reference-path data/reference/reference_dataset.parquet
    python scripts/run_monitoring.py --current-path  data/features/churn_features.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parents[1]))


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    reference_path = args.reference_path or config["paths"]["reference_data"]
    current_path   = args.current_path   or config["paths"]["features_data"]

    # ── Load datasets ─────────────────────────────────────────────
    if not Path(reference_path).exists():
        logger.error(f"Reference data not found: {reference_path}")
        logger.info("Run the full pipeline first: python scripts/run_pipeline.py")
        sys.exit(1)

    if not Path(current_path).exists():
        logger.error(f"Current data not found: {current_path}")
        sys.exit(1)

    reference_df = pd.read_parquet(reference_path)
    current_df   = pd.read_parquet(current_path)

    logger.info(f"Reference : {reference_df.shape}")
    logger.info(f"Current   : {current_df.shape}")

    # ── Run monitoring ────────────────────────────────────────────
    from src.monitoring.drift_monitor import DriftMonitor

    monitor = DriftMonitor(config)

    target = config["data"]["target_column"]
    y_true = current_df[target].astype(int) if target in current_df.columns else None

    # Simulate predictions for demo (in prod: load from prediction_log table)
    y_pred = None
    if y_true is not None and args.simulate_predictions:
        import numpy as np
        np.random.seed(42)
        y_pred = pd.Series(
            np.random.randint(0, 2, len(current_df)),
            name="prediction"
        )

    report = monitor.run_full_report(
        reference_df   = reference_df,
        current_df     = current_df,
        y_true_current = y_true,
        y_pred_current = y_pred,
    )

    output_path = args.output or "artifacts/reports/latest_drift_report.json"
    monitor.save_report(report, output_path)

    # ── Print summary ─────────────────────────────────────────────
    ds = report.get("drift_summary", {})
    ts = report.get("test_results",  {})
    print("\n" + "="*55)
    print("  DRIFT MONITORING REPORT")
    print("="*55)
    print(f"  Timestamp         : {report['timestamp']}")
    print(f"  Dataset Drift     : {'YES ⚠️' if ds.get('dataset_drift') else 'NO ✅'}")
    print(f"  Drifted Features  : {ds.get('n_drifted_features', 0)} / {ds.get('n_features', 0)}")
    print(f"  Share Drifted     : {ds.get('share_drifted', 0):.1%}")
    print(f"  Tests Passed      : {'ALL ✅' if ts.get('all_passed') else f'{ts.get(\"n_failed\",0)} FAILED ❌'}")

    if report.get("alerts"):
        print(f"\n  🚨 ALERTS ({len(report['alerts'])})")
        for a in report["alerts"]:
            icon = {"CRITICAL": "🔴", "WARNING": "🟡"}.get(a["level"], "⚪")
            print(f"  {icon} [{a['level']}] {a['msg']}")
    else:
        print("\n  ✅ No alerts")

    print("="*55)
    print(f"\n  Report saved → {output_path}")
    print(f"  Dashboard  → streamlit run dashboard/streamlit_app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drift monitoring")
    parser.add_argument("--config",               default="configs/config.yaml")
    parser.add_argument("--reference-path",        default=None)
    parser.add_argument("--current-path",          default=None)
    parser.add_argument("--output",                default=None)
    parser.add_argument("--simulate-predictions",  action="store_true",
                        help="Simulate model predictions for performance metrics")
    args = parser.parse_args()
    main(args)

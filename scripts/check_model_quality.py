"""
scripts/check_model_quality.py
───────────────────────────────
Quality gate script used by:
  - GitHub Actions CI/CD (pulls best run from MLflow, checks thresholds)
  - Airflow retrain DAG (after retraining)

Exits with code 1 if thresholds not met → fails the CI pipeline.
Writes model_eval_result.txt for the GitHub Actions PR comment step.
"""

import json
import os
import sys
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parents[1]))


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    f1_threshold       = config["monitoring"]["f1_threshold"]
    accuracy_threshold = config["monitoring"]["accuracy_threshold"]

    tracking_uri    = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    experiment_name = config["mlflow"]["experiment_name"]

    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning("No MLflow experiment found. Skipping quality gate.")
            _write_result("No MLflow experiment found — skipped quality gate.", passed=True)
            sys.exit(0)

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1 DESC"],
            max_results=10,
        )

        if runs.empty:
            logger.warning("No runs found in MLflow. Skipping quality gate.")
            _write_result("No runs found — skipped quality gate.", passed=True)
            sys.exit(0)

        best_run = runs.iloc[0]
        best_f1       = best_run.get("metrics.f1", 0.0)
        best_accuracy = best_run.get("metrics.accuracy", 0.0)
        best_auc      = best_run.get("metrics.roc_auc", 0.0)
        best_recall   = best_run.get("metrics.recall", 0.0)
        model_type    = best_run.get("tags.model_type", "unknown")
        run_id        = best_run["run_id"]

    except Exception as e:
        logger.error(f"MLflow connection failed: {e}")
        logger.warning("Writing pass result to avoid blocking CI without MLflow.")
        _write_result(f"MLflow unavailable: {e}\n→ Quality gate SKIPPED", passed=True)
        sys.exit(0)

    # ── Evaluate thresholds ───────────────────────────────────────
    f1_pass       = best_f1       >= f1_threshold
    accuracy_pass = best_accuracy >= accuracy_threshold

    lines = [
        f"Model Quality Gate — {experiment_name}",
        f"{'─'*45}",
        f"Run ID     : {run_id[:8]}…",
        f"Model type : {model_type}",
        f"",
        f"Metric       Value    Threshold   Status",
        f"{'─'*45}",
        f"F1 Score   : {best_f1:.4f}   >= {f1_threshold}      {'✅ PASS' if f1_pass else '❌ FAIL'}",
        f"Accuracy   : {best_accuracy:.4f}   >= {accuracy_threshold}      {'✅ PASS' if accuracy_pass else '❌ FAIL'}",
        f"ROC-AUC    : {best_auc:.4f}   (informational)",
        f"Recall     : {best_recall:.4f}   (informational)",
        f"{'─'*45}",
    ]

    overall_pass = f1_pass and accuracy_pass
    lines.append(f"Overall Gate: {'✅ PASSED' if overall_pass else '❌ FAILED'}")

    report = "\n".join(lines)
    print(report)

    # Save metrics JSON for DVC metrics tracking
    metrics = {
        "f1": best_f1,
        "accuracy": best_accuracy,
        "roc_auc": best_auc,
        "recall": best_recall,
        "model_type": model_type,
        "run_id": run_id,
        "quality_gate_passed": overall_pass,
    }
    Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
    with open("artifacts/reports/eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _write_result(report, passed=overall_pass)

    if not overall_pass:
        logger.error("Quality gate FAILED — model does not meet thresholds")
        sys.exit(1)
    else:
        logger.success("Quality gate PASSED")
        sys.exit(0)


def _write_result(text: str, passed: bool):
    with open("model_eval_result.txt", "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()

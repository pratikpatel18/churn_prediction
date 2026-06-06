"""
Monitoring Module — Evidently AI
──────────────────────────────────
Detects data drift and model degradation between a reference dataset
(what the model was trained on) and current production data.

Outputs:
  - JSON report consumed by the Streamlit dashboard
  - HTML report saved as an artifact
  - Alerts logged when drift exceeds thresholds
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationPreset,
    TargetDriftPreset,
)
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric,
    ClassificationQualityMetric,
)
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnDrift,
)


class DriftMonitor:
    """
    Runs Evidently drift and quality reports comparing reference vs current data.

    Usage:
        monitor = DriftMonitor(config)
        report  = monitor.run_full_report(reference_df, current_df, predictions_df)
        monitor.save_report(report, "artifacts/reports/drift_report.json")
    """

    def __init__(self, config: dict):
        self.config      = config
        self.target_col  = config["data"]["target_column"]
        self.threshold   = config["monitoring"]["drift_p_value"]
        self.reports_dir = Path(config["paths"]["reports"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def run_full_report(
        self,
        reference_df:   pd.DataFrame,
        current_df:     pd.DataFrame,
        y_true_current: Optional[pd.Series] = None,
        y_pred_current: Optional[pd.Series] = None,
    ) -> dict:
        """
        Runs all Evidently metrics. Returns a structured dict for the dashboard.
        """
        timestamp = datetime.utcnow().isoformat()

        col_mapping = self._build_column_mapping(reference_df, y_true_current)

        # Prepare DataFrames with prediction columns if available
        ref_eval = reference_df.copy()
        cur_eval = current_df.copy()
        if y_pred_current is not None:
            cur_eval["prediction"] = y_pred_current.values
        if y_true_current is not None:
            cur_eval[self.target_col] = y_true_current.values

        # ── 1. Data Drift Report ───────────────────────────────────
        drift_report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        drift_report.run(
            reference_data = reference_df,
            current_data   = current_df,
            column_mapping = col_mapping,
        )

        # ── 2. Model Performance Report (if labels available) ──────
        perf_report = None
        if y_pred_current is not None and y_true_current is not None:
            perf_report = Report(metrics=[ClassificationPreset()])
            perf_report.run(
                reference_data = ref_eval if "prediction" in ref_eval.columns else None,
                current_data   = cur_eval,
                column_mapping = col_mapping,
            )

        # ── 3. Extract key metrics for dashboard ───────────────────
        drift_dict   = drift_report.as_dict()
        drift_summary = self._extract_drift_summary(drift_dict)

        perf_summary = {}
        if perf_report:
            perf_dict    = perf_report.as_dict()
            perf_summary = self._extract_perf_summary(perf_dict)

        # ── 4. Save HTML reports ───────────────────────────────────
        drift_html = self.reports_dir / f"drift_{timestamp[:10]}.html"
        drift_report.save_html(str(drift_html))
        logger.info(f"Drift HTML report → {drift_html}")

        if perf_report:
            perf_html = self.reports_dir / f"performance_{timestamp[:10]}.html"
            perf_report.save_html(str(perf_html))

        # ── 5. Run Test Suite ──────────────────────────────────────
        test_results = self._run_test_suite(reference_df, current_df, col_mapping)

        # ── 6. Assemble final result dict ──────────────────────────
        result = {
            "timestamp":     timestamp,
            "drift_summary": drift_summary,
            "perf_summary":  perf_summary,
            "test_results":  test_results,
            "alerts":        self._generate_alerts(drift_summary, perf_summary),
        }

        logger.info(
            f"Monitoring complete. "
            f"Dataset drift: {drift_summary.get('dataset_drift', False)} | "
            f"Drifted cols: {drift_summary.get('n_drifted_features', 0)}"
        )
        return result

    def save_report(self, report: dict, path: str) -> None:
        """Persist JSON report for the Streamlit dashboard to load."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.success(f"Report saved → {path}")

    def run_test_suite_only(
        self,
        reference_df: pd.DataFrame,
        current_df:   pd.DataFrame,
    ) -> bool:
        """Quick pass/fail check — used by the retraining DAG."""
        col_mapping = self._build_column_mapping(reference_df)
        results = self._run_test_suite(reference_df, current_df, col_mapping)
        return results["all_passed"]

    # ──────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────

    def _build_column_mapping(
        self,
        df:           pd.DataFrame,
        y:            Optional[pd.Series] = None,
    ) -> ColumnMapping:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # Remove target from feature lists
        for lst in [num_cols, cat_cols]:
            if self.target_col in lst:
                lst.remove(self.target_col)

        return ColumnMapping(
            target            = self.target_col if self.target_col in df.columns else None,
            prediction        = "prediction" if "prediction" in df.columns else None,
            numerical_features  = num_cols,
            categorical_features = cat_cols,
        )

    def _run_test_suite(
        self,
        reference_df: pd.DataFrame,
        current_df:   pd.DataFrame,
        col_mapping:  ColumnMapping,
    ) -> dict:
        suite = TestSuite(tests=[
            DataDriftTestPreset(),
            DataQualityTestPreset(),
            TestNumberOfDriftedColumns(lte=5),
            TestShareOfDriftedColumns(lte=0.3),
        ])
        suite.run(
            reference_data = reference_df,
            current_data   = current_df,
            column_mapping = col_mapping,
        )
        result_dict = suite.as_dict()
        tests       = result_dict.get("tests", [])
        all_passed  = all(t.get("status") == "SUCCESS" for t in tests)

        return {
            "all_passed":   all_passed,
            "n_tests":      len(tests),
            "n_failed":     sum(1 for t in tests if t.get("status") != "SUCCESS"),
            "test_details": [
                {
                    "name":   t.get("name"),
                    "status": t.get("status"),
                    "detail": t.get("description", ""),
                }
                for t in tests
            ],
        }

    @staticmethod
    def _extract_drift_summary(drift_dict: dict) -> dict:
        """Pull out key drift metrics from Evidently report dict."""
        try:
            metrics = drift_dict.get("metrics", [])
            drift_metric = next(
                (m for m in metrics if "DatasetDriftMetric" in str(m.get("metric", ""))),
                {}
            )
            result = drift_metric.get("result", {})
            return {
                "dataset_drift":      result.get("dataset_drift", False),
                "n_drifted_features": result.get("number_of_drifted_columns", 0),
                "n_features":         result.get("number_of_columns", 0),
                "share_drifted":      result.get("share_of_drifted_columns", 0.0),
            }
        except Exception:
            return {}

    @staticmethod
    def _extract_perf_summary(perf_dict: dict) -> dict:
        """Pull classification metrics from Evidently performance report."""
        try:
            metrics = perf_dict.get("metrics", [])
            qual_metric = next(
                (m for m in metrics if "ClassificationQualityMetric" in str(m.get("metric", ""))),
                {}
            )
            current = qual_metric.get("result", {}).get("current", {})
            return {
                "accuracy":  current.get("accuracy", None),
                "f1":        current.get("f1",        None),
                "precision": current.get("precision", None),
                "recall":    current.get("recall",    None),
                "roc_auc":   current.get("roc_auc",   None),
            }
        except Exception:
            return {}

    def _generate_alerts(self, drift_summary: dict, perf_summary: dict) -> list[dict]:
        alerts = []

        if drift_summary.get("dataset_drift"):
            alerts.append({
                "level": "WARNING",
                "type":  "data_drift",
                "msg":   (
                    f"Dataset drift detected: "
                    f"{drift_summary.get('n_drifted_features')} / "
                    f"{drift_summary.get('n_features')} features drifted "
                    f"({drift_summary.get('share_drifted', 0):.0%})"
                ),
            })

        f1 = perf_summary.get("f1")
        if f1 is not None and f1 < self.config["monitoring"]["f1_threshold"]:
            alerts.append({
                "level": "CRITICAL",
                "type":  "model_degradation",
                "msg":   (
                    f"Model F1 ({f1:.3f}) below threshold "
                    f"({self.config['monitoring']['f1_threshold']}). "
                    f"Retraining recommended."
                ),
            })

        return alerts

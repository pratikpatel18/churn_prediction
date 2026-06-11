"""
Data Validation Module — Great Expectations
────────────────────────────────────────────
Runs a suite of automated quality checks on raw and processed data.
Any pipeline step can call `run_validation()` before proceeding.
If validation fails the pipeline raises an exception — this prevents
bad data from silently polluting your model.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from great_expectations.checkpoint import SimpleCheckpoint
from great_expectations.core.batch import RuntimeBatchRequest
from loguru import logger

# Great Expectations
import great_expectations as gx


class DataValidator:
    """
    Wraps Great Expectations to validate DataFrames in-pipeline.

    Usage:
        validator = DataValidator()
        result = validator.validate_raw(df)
        if not result["success"]:
            raise ValueError("Raw data failed validation!")
    """

    def __init__(self, ge_root_dir: str = "great_expectations"):
        self.ge_root_dir = Path(ge_root_dir)
        self._context = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def validate_raw(self, df: pd.DataFrame) -> dict:
        """Validate raw ingested data against the raw expectation suite."""
        logger.info("Running raw data validation suite …")
        return self._run_suite(df, suite_name="raw_data_suite")

    def validate_features(self, df: pd.DataFrame) -> dict:
        """Validate engineered feature DataFrame."""
        logger.info("Running feature validation suite …")
        return self._run_suite(df, suite_name="feature_suite")

    def build_raw_expectation_suite(self, df: pd.DataFrame) -> None:
        """
        One-time setup: build expectations from a sample DataFrame.
        Run this interactively once to generate the suite JSON.
        """
        context = self._get_context()

        suite_name = "raw_data_suite"
        suite = context.add_or_update_expectation_suite(suite_name)

        validator = context.get_validator(
            batch_request=self._make_batch_request(df, suite_name),
            expectation_suite_name=suite_name,
        )

        # ── Schema checks ──────────────────────────────────────────
        validator.expect_table_columns_to_match_ordered_list(
            column_list=[
                "customerID",
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
                "Churn",
            ]
        )
        validator.expect_table_row_count_to_be_between(min_value=100, max_value=100_000)

        # ── Completeness checks ────────────────────────────────────
        for col in ["customerID", "gender", "tenure", "MonthlyCharges", "Churn"]:
            validator.expect_column_values_to_not_be_null(col)

        # ── Value range checks ─────────────────────────────────────
        validator.expect_column_values_to_be_between(
            "tenure", min_value=0, max_value=72
        )
        validator.expect_column_values_to_be_between(
            "MonthlyCharges", min_value=0, max_value=200
        )
        validator.expect_column_values_to_be_between(
            "TotalCharges", min_value=0, max_value=10_000
        )
        validator.expect_column_values_to_be_between(
            "SeniorCitizen", min_value=0, max_value=1
        )

        # ── Cardinality / set checks ───────────────────────────────
        validator.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
        validator.expect_column_values_to_be_in_set("Churn", [0, 1])
        validator.expect_column_values_to_be_in_set(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        validator.expect_column_values_to_be_in_set(
            "InternetService", ["DSL", "Fiber optic", "No"]
        )

        # ── Uniqueness ─────────────────────────────────────────────
        validator.expect_column_values_to_be_unique("customerID")

        # ── Statistical checks ─────────────────────────────────────
        # Churn rate should be between 5%–50% (sanity check)
        validator.expect_column_mean_to_be_between(
            "Churn", min_value=0.05, max_value=0.50
        )

        validator.save_expectation_suite(discard_failed_expectations=False)
        logger.success(f"Expectation suite '{suite_name}' saved.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_suite(self, df: pd.DataFrame, suite_name: str) -> dict:
        context = self._get_context()

        # Programmatic validation (no disk I/O needed)
        validator = context.get_validator(
            batch_request=self._make_batch_request(df, suite_name),
            expectation_suite_name=suite_name,
        )

        results = validator.validate()
        success = results["success"]
        n_failed = sum(1 for r in results["results"] if not r["success"])
        n_total = len(results["results"])

        if success:
            logger.success(f"Validation passed: {n_total}/{n_total} expectations met")
        else:
            logger.error(f"Validation FAILED: {n_failed}/{n_total} expectations failed")
            for r in results["results"]:
                if not r["success"]:
                    logger.error(
                        f"  ✗ {r['expectation_config']['expectation_type']} — "
                        f"{r['expectation_config']['kwargs']}"
                    )

        return {
            "success": success,
            "n_total": n_total,
            "n_failed": n_failed,
            "details": results,
        }

    def _get_context(self):
        """Lazy-init Great Expectations DataContext."""
        if self._context is None:
            try:
                self._context = gx.get_context(context_root_dir=str(self.ge_root_dir))
            except Exception:
                # Create an ephemeral in-memory context if no project exists
                self._context = gx.get_context()
        return self._context

    def _make_batch_request(self, df: pd.DataFrame, name: str) -> RuntimeBatchRequest:
        return RuntimeBatchRequest(
            datasource_name="runtime_datasource",
            data_connector_name="default_runtime_data_connector_name",
            data_asset_name=name,
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_identifier"},
        )


# ──────────────────────────────────────────────────────────────────────
# Standalone validation utility (for quick checks in Airflow tasks)
# ──────────────────────────────────────────────────────────────────────


def validate_dataframe(df: pd.DataFrame, suite: str = "raw_data_suite") -> bool:
    """
    Quick helper used by Airflow tasks.
    Returns True/False and logs failures.
    """
    validator = DataValidator()
    result = validator._run_suite(df, suite)
    return result["success"]

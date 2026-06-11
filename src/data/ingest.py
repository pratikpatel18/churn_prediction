"""
Data Ingestion Module
─────────────────────
Handles raw data loading, initial cleaning, and type casting.
Supports CSV, Parquet, and PostgreSQL sources.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine


class DataIngester:
    """
    Loads raw customer data from various sources,
    performs initial cleaning and schema enforcement.
    """

    EXPECTED_COLUMNS = [
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

    def __init__(self, config: dict):
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest_csv(self, filepath: str) -> pd.DataFrame:
        """Load from CSV and run initial cleaning."""
        logger.info(f"Ingesting CSV from: {filepath}")
        df = pd.read_csv(filepath)
        df = self._initial_clean(df)
        logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")
        return df

    def ingest_parquet(self, filepath: str) -> pd.DataFrame:
        """Load from Parquet file."""
        logger.info(f"Ingesting Parquet from: {filepath}")
        df = pd.read_parquet(filepath)
        df = self._initial_clean(df)
        logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")
        return df

    def ingest_from_db(self, table: str = "raw_customers") -> pd.DataFrame:
        """Pull raw data from PostgreSQL (simulates production ingestion)."""
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            raise EnvironmentError("DATABASE_URL not set in environment")
        engine = create_engine(db_url)
        logger.info(f"Ingesting from DB table: {table}")
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        df = self._initial_clean(df)
        logger.info(f"Loaded {len(df)} rows from database")
        return df

    def save_processed(self, df: pd.DataFrame, output_path: str) -> None:
        """Persist cleaned data as Parquet for downstream steps."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.success(f"Saved processed data → {output_path}  ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initial_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Schema enforcement, type coercion, and basic cleaning."""
        df = df.copy()

        # 1. Column validation
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")

        # 2. TotalCharges often arrives as string with spaces → coerce to float
        if df["TotalCharges"].dtype == object:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # 3. Fill NaN TotalCharges with MonthlyCharges (new customers with tenure=0)
        mask = df["TotalCharges"].isna()
        df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"]
        logger.info(f"Imputed {mask.sum()} missing TotalCharges values")

        # 4. Standardise Churn column: 'Yes'/'No' → 1/0
        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

        # 5. Strip whitespace from string columns
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()

        # 6. SeniorCitizen: keep as int (already 0/1)
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

        # 7. Drop exact duplicate rows
        n_before = len(df)
        df = df.drop_duplicates(subset=["customerID"])
        n_dropped = n_before - len(df)
        if n_dropped:
            logger.warning(f"Dropped {n_dropped} duplicate customerIDs")

        # 8. Reset index
        df = df.reset_index(drop=True)

        logger.info("Initial cleaning complete")
        return df


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Helper to load YAML config."""
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)

"""
Inference Module
────────────────
Loads the Production model from MLflow Registry and serves predictions.
Used by FastAPI and the monitoring retrain script.
"""

import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from loguru import logger


class ChurnPredictor:
    """
    Thin wrapper around the MLflow production model.
    Handles model loading, preprocessing, and prediction.
    """

    def __init__(self, config: dict):
        self.config     = config
        self.model_name = config["mlflow"]["model_name"]
        self.threshold  = config["api"].get("prediction_threshold", 0.5)
        self.model      = None
        self._load_model()

    def _load_model(self, stage: str = "Production") -> None:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            self.config["mlflow"]["tracking_uri"]
        )
        mlflow.set_tracking_uri(tracking_uri)

        # Try loading by stage first
        model_uri = f"models:/{self.model_name}/{stage}"
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            logger.success(f"Loaded model: {model_uri}")
            return
        except Exception as e:
            logger.warning(f"Could not load '{stage}' model: {e}")

        # Fallback: find best run and load directly from run artifacts
        try:
            logger.info("Trying direct run artifact load...")
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            versions = client.search_model_versions(
                f"name='{self.model_name}'"
            )
            if versions:
                # Get the most recent version
                latest = sorted(versions, key=lambda v: int(v.version))[-1]
                run_uri = f"runs:/{latest.run_id}/model"
                logger.info(f"Loading from run: {run_uri}")
                self.model = mlflow.pyfunc.load_model(run_uri)
                logger.success(f"Model loaded from run artifact: {run_uri}")
            else:
                logger.warning("No model versions found in registry")
                self.model = None
        except Exception as e2:
            logger.warning(f"Could not load model from run: {e2}")
            self.model = None

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Returns prediction dict:
        {
          "churn_probability": [0.87, 0.12, ...],
          "churn_prediction":  [1, 0, ...],
          "risk_label":        ["High", "Low", ...]
        }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        probs = self.model.predict(df)

        # pyfunc may return raw logits for some flavours — normalise
        if probs.ndim == 2:
            probs = probs[:, 1]

        preds = (probs >= self.threshold).astype(int)
        labels = self._risk_labels(probs)

        return {
            "churn_probability": probs.tolist(),
            "churn_prediction":  preds.tolist(),
            "risk_label":        labels,
        }

    @staticmethod
    def _risk_labels(probs: np.ndarray) -> list[str]:
        labels = []
        for p in probs:
            if p >= 0.75:
                labels.append("Critical")
            elif p >= 0.50:
                labels.append("High")
            elif p >= 0.30:
                labels.append("Medium")
            else:
                labels.append("Low")
        return labels

    def predict_single(self, features: dict) -> dict:
        """Convenience wrapper for single-record REST predictions."""
        df = pd.DataFrame([features])
        result = self.predict(df)
        return {
            "churn_probability": result["churn_probability"][0],
            "churn_prediction":  result["churn_prediction"][0],
            "risk_label":        result["risk_label"][0],
        }

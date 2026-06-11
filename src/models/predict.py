"""
Inference Module
"""

import os
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from loguru import logger


class ChurnPredictor:
    def __init__(self, config: dict):
        self.config     = config
        self.model_name = config["mlflow"]["model_name"]
        self.threshold  = config["api"].get("prediction_threshold", 0.5)
        self.model      = None
        self._lgbm_model = None
        self._load_model()

    def _load_model(self, stage: str = "Production") -> None:
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            self.config["mlflow"]["tracking_uri"]
        )
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = f"models:/{self.model_name}/{stage}"
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            logger.success(f"Loaded model: {model_uri}")
        except Exception as e:
            logger.warning(f"Could not load '{stage}' model: {e}")
            try:
                logger.info("Trying direct run artifact load...")
                client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                versions = client.search_model_versions(f"name='{self.model_name}'")
                if versions:
                    latest = sorted(versions, key=lambda v: int(v.version))[-1]
                    run_uri = f"runs:/{latest.run_id}/model"
                    logger.info(f"Loading from run: {run_uri}")
                    self.model = mlflow.pyfunc.load_model(run_uri)
                    logger.success(f"Model loaded from run artifact: {run_uri}")
                else:
                    logger.warning("No model versions found in registry")
                    self.model = None
                    return
            except Exception as e2:
                logger.warning(f"Could not load model from run: {e2}")
                self.model = None
                return

        # Unwrap the underlying LightGBM model for predict_proba
        try:
            unwrapped = self.model.unwrap_python_model()
            self._lgbm_model = unwrapped
            logger.success("Unwrapped python model for predict_proba")
        except Exception:
            pass

        # Try accessing the lgb booster directly
        if self._lgbm_model is None:
            try:
                self._lgbm_model = self.model._model_impl.lgb_model
                logger.success("Accessed lgb_model directly")
            except Exception:
                pass

        if self._lgbm_model is None:
            logger.warning("Could not unwrap LightGBM model — will use pyfunc predict")

    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        probs = None

        # Try predict_proba on unwrapped model
        if self._lgbm_model is not None:
            try:
                raw = self._lgbm_model.predict_proba(df)
                if raw.ndim == 2:
                    probs = raw[:, 1]
                else:
                    probs = raw
                logger.debug(f"predict_proba result: {probs}")
            except Exception as e:
                logger.warning(f"predict_proba failed: {e}")
                probs = None

        # Fallback: try pyfunc predict and check if it looks like probabilities
        if probs is None:
            raw = self.model.predict(df)
            if raw.ndim == 2:
                probs = raw[:, 1]
            elif np.all((raw >= 0) & (raw <= 1)) and not np.all(raw == raw.astype(int)):
                probs = raw  # already probabilities
            else:
                # It returned class labels — we cannot get probabilities this way
                logger.warning("pyfunc returned class labels not probabilities — retraining recommended")
                probs = raw.astype(float)

        preds  = (probs >= self.threshold).astype(int)
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
        df = pd.DataFrame([features])
        result = self.predict(df)
        return {
            "churn_probability": result["churn_probability"][0],
            "churn_prediction":  result["churn_prediction"][0],
            "risk_label":        result["risk_label"][0],
        }

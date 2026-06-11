import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field, validator

from src.features.engineer import FeatureEngineer
from src.models.predict import ChurnPredictor


def _load_config():
    path = os.getenv("CONFIG_PATH", "configs/config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


CONFIG = _load_config()
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", CONFIG["mlflow"]["tracking_uri"])
CONFIG["mlflow"]["tracking_uri"] = MLFLOW_URI

predictor = None
engineer = None
_metrics = {
    "total_predictions": 0,
    "total_churners": 0,
    "avg_latency_ms": 0.0,
    "last_prediction_at": None,
}


def _load_preprocessor(eng):
    paths = [
        "artifacts/models/preprocessor.joblib",
        "/app/artifacts/models/preprocessor.joblib",
        "/opt/airflow/artifacts/models/preprocessor.joblib",
    ]
    for path in paths:
        try:
            eng.preprocessor = joblib.load(path)
            scaler_path = path.replace("preprocessor", "scaler")
            eng.scaler = joblib.load(scaler_path)
            logger.success(f"Preprocessor loaded from: {path}")
            return True
        except Exception:
            continue
    logger.warning("Preprocessor not found in any path")
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, engineer
    logger.info(f"Connecting to MLflow at: {MLFLOW_URI}")

    try:
        predictor = ChurnPredictor(CONFIG)
        logger.success("Model loaded")
    except Exception as e:
        logger.warning(f"Model not loaded: {e}")
        predictor = None

    try:
        engineer = FeatureEngineer(CONFIG)
        _load_preprocessor(engineer)
    except Exception as e:
        logger.warning(f"Engineer error: {e}")
        engineer = None

    logger.info("API startup complete")
    yield
    logger.info("API shutting down")


app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    @validator("Contract")
    def validate_contract(cls, v):
        allowed = ["Month-to-month", "One year", "Two year"]
        if v not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v

    @validator("InternetService")
    def validate_internet(cls, v):
        allowed = ["DSL", "Fiber optic", "No"]
        if v not in allowed:
            raise ValueError(f"Must be one of {allowed}")
        return v


class PredictionResponse(BaseModel):
    request_id: str
    customer_id: str | None = None
    churn_probability: float
    churn_prediction: int
    risk_label: str
    timestamp: str


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures]
    customer_ids: list[str] | None = None


class BatchResponse(BaseModel):
    request_id: str
    predictions: list[PredictionResponse]
    n_processed: int
    latency_ms: float


@app.get("/health")
async def health():
    model_ok = predictor is not None and predictor.model is not None
    return {
        "status": "ok",
        "model_loaded": model_ok,
        "mlflow_uri": MLFLOW_URI,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model/info")
async def model_info():
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_URI)
        versions = client.search_model_versions(f"name='{CONFIG['mlflow']['model_name']}'")
        return {
            "model_name": CONFIG["mlflow"]["model_name"],
            "versions": [
                {"version": v.version, "stage": v.current_stage, "run_id": v.run_id}
                for v in versions
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerFeatures):
    global engineer

    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")

    if engineer is None:
        engineer = FeatureEngineer(CONFIG)

    if not hasattr(engineer, "preprocessor") or engineer.preprocessor is None:
        _load_preprocessor(engineer)

    t0 = time.perf_counter()
    rid = str(uuid.uuid4())
    try:
        raw_df = pd.DataFrame([customer.dict()])
        feat_df = engineer.transform(raw_df)
        feat_df = feat_df.drop(columns=["Churn"], errors="ignore")
        result = predictor.predict(feat_df)
        lat = (time.perf_counter() - t0) * 1000
        _update_metrics(result["churn_prediction"][0], lat)
        return PredictionResponse(
            request_id=rid,
            churn_probability=round(result["churn_probability"][0], 4),
            churn_prediction=result["churn_prediction"][0],
            risk_label=result["risk_label"][0],
            timestamp=datetime.utcnow().isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(payload: BatchRequest):
    if not predictor or not predictor.model:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not payload.customers:
        raise HTTPException(status_code=400, detail="Empty customers list")

    t0 = time.perf_counter()
    rid = str(uuid.uuid4())
    try:
        raw_df = pd.DataFrame([c.dict() for c in payload.customers])
        feat_df = engineer.transform(raw_df)
        feat_df = feat_df.drop(columns=["Churn"], errors="ignore")
        result = predictor.predict(feat_df)
        ids = payload.customer_ids or [None] * len(payload.customers)
        preds = [
            PredictionResponse(
                request_id=rid,
                customer_id=ids[i],
                churn_probability=round(result["churn_probability"][i], 4),
                churn_prediction=result["churn_prediction"][i],
                risk_label=result["risk_label"][i],
                timestamp=datetime.utcnow().isoformat(),
            )
            for i in range(len(payload.customers))
        ]
        lat = (time.perf_counter() - t0) * 1000
        return BatchResponse(
            request_id=rid,
            predictions=preds,
            n_processed=len(preds),
            latency_ms=round(lat, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    return _metrics


def _update_metrics(pred, lat):
    n = _metrics["total_predictions"] + 1
    _metrics["total_predictions"] = n
    _metrics["total_churners"] += pred
    _metrics["avg_latency_ms"] = (_metrics["avg_latency_ms"] * (n - 1) + lat) / n
    _metrics["last_prediction_at"] = datetime.utcnow().isoformat()

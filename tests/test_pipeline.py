"""
Test Suite — Churn Prediction Pipeline
───────────────────────────────────────
Covers: data ingestion, feature engineering, model training, and API.
Run: pytest tests/ -v --cov=src
"""

import json
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1]))
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def raw_df():
    """Minimal synthetic Telco churn dataset for tests."""
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "customerID":        [f"C{i:04d}" for i in range(n)],
        "gender":            np.random.choice(["Male", "Female"], n),
        "SeniorCitizen":     np.random.randint(0, 2, n),
        "Partner":           np.random.choice(["Yes", "No"], n),
        "Dependents":        np.random.choice(["Yes", "No"], n),
        "tenure":            np.random.randint(0, 72, n),
        "PhoneService":      np.random.choice(["Yes", "No"], n),
        "MultipleLines":     np.random.choice(["Yes", "No", "No phone service"], n),
        "InternetService":   np.random.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity":    np.random.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup":      np.random.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection":  np.random.choice(["Yes", "No", "No internet service"], n),
        "TechSupport":       np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV":       np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies":   np.random.choice(["Yes", "No", "No internet service"], n),
        "Contract":          np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling":  np.random.choice(["Yes", "No"], n),
        "PaymentMethod":     np.random.choice([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], n),
        "MonthlyCharges":    np.round(np.random.uniform(20, 100, n), 2),
        "TotalCharges":      np.round(np.random.uniform(20, 8000, n), 2),
        "Churn":             np.random.choice(["Yes", "No"], n, p=[0.27, 0.73]),
    })


@pytest.fixture(scope="session")
def cleaned_df(raw_df, config):
    from src.data.ingest import DataIngester
    ingester = DataIngester(config)
    return ingester._initial_clean(raw_df.copy())


@pytest.fixture(scope="session")
def features_df(cleaned_df, config, tmp_path_factory):
    from src.features.engineer import FeatureEngineer
    artifact_dir = str(tmp_path_factory.mktemp("artifacts"))
    engineer = FeatureEngineer(config=config, artifact_dir=artifact_dir)
    return engineer.fit_transform(cleaned_df)


# ─────────────────────────────────────────────────────────────────────
# 1. Data Ingestion Tests
# ─────────────────────────────────────────────────────────────────────

class TestDataIngester:

    def test_initial_clean_returns_dataframe(self, cleaned_df):
        assert isinstance(cleaned_df, pd.DataFrame)

    def test_churn_column_is_binary(self, cleaned_df):
        assert set(cleaned_df["Churn"].unique()).issubset({0, 1})

    def test_no_duplicate_customer_ids(self, cleaned_df):
        assert cleaned_df["customerID"].nunique() == len(cleaned_df)

    def test_total_charges_no_nulls(self, cleaned_df):
        assert cleaned_df["TotalCharges"].isna().sum() == 0

    def test_all_expected_columns_present(self, cleaned_df):
        from src.data.ingest import DataIngester
        expected = set(DataIngester.EXPECTED_COLUMNS)
        assert expected.issubset(set(cleaned_df.columns))

    def test_senior_citizen_is_binary(self, cleaned_df):
        assert set(cleaned_df["SeniorCitizen"].unique()).issubset({0, 1})

    def test_monthly_charges_positive(self, cleaned_df):
        assert (cleaned_df["MonthlyCharges"] >= 0).all()

    def test_tenure_in_valid_range(self, cleaned_df):
        assert cleaned_df["tenure"].between(0, 72).all()

    def test_string_columns_stripped(self, cleaned_df):
        for col in ["gender", "Contract"]:
            assert not cleaned_df[col].str.startswith(" ").any()

    def test_missing_column_raises_error(self, config):
        from src.data.ingest import DataIngester
        ingester = DataIngester(config)
        bad_df = pd.DataFrame({"col1": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing expected columns"):
            ingester._initial_clean(bad_df)


# ─────────────────────────────────────────────────────────────────────
# 2. Feature Engineering Tests
# ─────────────────────────────────────────────────────────────────────

class TestFeatureEngineer:

    def test_returns_dataframe(self, features_df):
        assert isinstance(features_df, pd.DataFrame)

    def test_target_column_preserved(self, features_df, config):
        target = config["data"]["target_column"]
        assert target in features_df.columns

    def test_no_nulls_in_features(self, features_df, config):
        target = config["data"]["target_column"]
        X = features_df.drop(columns=[target])
        assert X.isna().sum().sum() == 0

    def test_tenure_group_created(self, features_df):
        # tenure_group gets one-hot encoded — check OHE columns exist
        ohe_cols = [c for c in features_df.columns if "tenure_group" in c]
        assert len(ohe_cols) > 0

    def test_risk_score_is_numeric(self, features_df):
        if "risk_score" in features_df.columns:
            assert pd.api.types.is_numeric_dtype(features_df["risk_score"])

    def test_charge_ratio_non_negative(self, features_df):
        if "charge_ratio" in features_df.columns:
            assert (features_df["charge_ratio"] >= 0).all()

    def test_services_count_non_negative(self, features_df):
        if "services_count" in features_df.columns:
            assert (features_df["services_count"] >= 0).all()

    def test_customer_id_dropped(self, features_df):
        assert "customerID" not in features_df.columns

    def test_feature_count_reasonable(self, features_df, config):
        n_features = features_df.shape[1] - 1  # subtract target
        assert 15 <= n_features <= 80, f"Unexpected feature count: {n_features}"

    def test_transform_matches_fit_transform_shape(self, cleaned_df, config, tmp_path):
        from src.features.engineer import FeatureEngineer
        engineer = FeatureEngineer(config=config, artifact_dir=str(tmp_path))
        df_fit = engineer.fit_transform(cleaned_df.copy())

        # transform on fresh slice should match column count
        df_new = cleaned_df.iloc[:10].copy()
        df_tf  = engineer.transform(df_new)
        assert df_tf.shape[1] == df_fit.shape[1]


# ─────────────────────────────────────────────────────────────────────
# 3. Model Metrics Computation Tests
# ─────────────────────────────────────────────────────────────────────

class TestMetricsComputation:

    def test_compute_metrics_perfect(self):
        from src.models.train import compute_metrics
        y = np.array([0, 0, 1, 1])
        m = compute_metrics(y, y, y.astype(float))
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0

    def test_compute_metrics_keys(self):
        from src.models.train import compute_metrics
        y    = np.array([0, 1, 0, 1])
        pred = np.array([0, 1, 1, 0])
        prob = np.array([0.1, 0.9, 0.6, 0.4])
        m = compute_metrics(y, pred, prob)
        assert set(m.keys()) == {"accuracy", "f1", "precision", "recall", "roc_auc"}

    def test_all_metrics_between_0_and_1(self):
        from src.models.train import compute_metrics
        np.random.seed(0)
        y    = np.random.randint(0, 2, 100)
        pred = np.random.randint(0, 2, 100)
        prob = np.random.uniform(0, 1, 100)
        m = compute_metrics(y, pred, prob)
        for k, v in m.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


# ─────────────────────────────────────────────────────────────────────
# 4. FastAPI Endpoint Tests
# ─────────────────────────────────────────────────────────────────────

class TestFastAPI:
    """
    Tests FastAPI endpoints by mocking the model and preprocessor
    so no MLflow server is needed.
    """

    @pytest.fixture(autouse=True)
    def mock_deps(self, config, monkeypatch):
        """Mock ChurnPredictor and FeatureEngineer for API tests."""

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            "churn_probability": [0.75],
            "churn_prediction":  [1],
            "risk_label":        ["High"],
        }

        mock_engineer = MagicMock()
        mock_engineer.transform.side_effect = lambda df: df.assign(Churn=0)

        monkeypatch.setattr("src.api.main.ChurnPredictor", lambda c: mock_predictor)
        monkeypatch.setattr("src.api.main.FeatureEngineer", lambda c, **kw: mock_engineer)

        import src.api.main as api_module
        api_module.predictor  = mock_predictor
        api_module.engineer   = mock_engineer
        api_module.CONFIG     = config

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    @pytest.fixture
    def sample_payload(self):
        return {
            "gender": "Male", "SeniorCitizen": 0,
            "Partner": "Yes", "Dependents": "No",
            "tenure": 12, "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "Fiber optic",
            "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No",
            "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": "Month-to-month", "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.5, "TotalCharges": 846.0,
        }

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_predict_endpoint_returns_200(self, client, sample_payload):
        r = client.post("/predict", json=sample_payload)
        assert r.status_code == 200

    def test_predict_response_schema(self, client, sample_payload):
        r = client.post("/predict", json=sample_payload)
        data = r.json()
        assert "churn_probability" in data
        assert "churn_prediction"  in data
        assert "risk_label"        in data
        assert "request_id"        in data

    def test_predict_probability_range(self, client, sample_payload):
        r = client.post("/predict", json=sample_payload)
        prob = r.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_invalid_contract_rejected(self, client, sample_payload):
        bad = {**sample_payload, "Contract": "InvalidContract"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422  # Pydantic validation error

    def test_predict_invalid_internet_rejected(self, client, sample_payload):
        bad = {**sample_payload, "InternetService": "Satellite"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_batch_endpoint(self, client, sample_payload):
        payload = {"customers": [sample_payload, sample_payload]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["n_processed"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty_rejected(self, client):
        r = client.post("/predict/batch", json={"customers": []})
        assert r.status_code == 400

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "total_predictions" in r.json()


# ─────────────────────────────────────────────────────────────────────
# 5. Risk Label Tests
# ─────────────────────────────────────────────────────────────────────

class TestRiskLabels:

    def test_critical_label(self):
        from src.models.predict import ChurnPredictor
        labels = ChurnPredictor._risk_labels(np.array([0.80, 0.76]))
        assert all(l == "Critical" for l in labels)

    def test_high_label(self):
        from src.models.predict import ChurnPredictor
        labels = ChurnPredictor._risk_labels(np.array([0.60, 0.51]))
        assert all(l == "High" for l in labels)

    def test_medium_label(self):
        from src.models.predict import ChurnPredictor
        labels = ChurnPredictor._risk_labels(np.array([0.40, 0.31]))
        assert all(l == "Medium" for l in labels)

    def test_low_label(self):
        from src.models.predict import ChurnPredictor
        labels = ChurnPredictor._risk_labels(np.array([0.10, 0.29]))
        assert all(l == "Low" for l in labels)

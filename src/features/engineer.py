import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler
import joblib


class FeatureEngineer:
    BINARY_MAP = {"Yes": 1, "No": 0, "No phone service": 0, "No internet service": 0}
    CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
    SERVICES_COLS = [
        "PhoneService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    def __init__(self, config, artifact_dir="artifacts/models"):
        self.config = config
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = None
        self.scaler = None

    def fit_transform(self, df):
        df = self._create_features(df)
        df = self._encode_and_scale(df, fit=True)
        return df

    def transform(self, df):
        df = self._create_features(df)
        df = self._encode_and_scale(df, fit=False)
        return df

    def save_preprocessor(self):
        path = self.artifact_dir / "preprocessor.joblib"
        joblib.dump(self.scaler, path)
        return str(path)

    def load_preprocessor(self, path=None):
        for p in [
            path,
            str(self.artifact_dir / "preprocessor.joblib"),
            "artifacts/models/preprocessor.joblib",
            "/app/artifacts/models/preprocessor.joblib",
        ]:
            if p is None:
                continue
            try:
                self.preprocessor = joblib.load(p)
                try:
                    self.scaler = joblib.load(str(p).replace("preprocessor", "scaler"))
                except Exception:
                    self.scaler = self.preprocessor
                logger.info(f"Preprocessor loaded from {p}")
                return
            except Exception:
                continue

    def _create_features(self, df):
        df = df.copy()
        cfg = self.config.get("feature_engineering", {})
        bins = cfg.get("tenure_bins", [0, 12, 24, 48, 72])
        labels = cfg.get("tenure_labels", ["new", "developing", "established", "loyal"])
        df["tenure_group"] = pd.cut(
            df["tenure"], bins=bins, labels=labels, include_lowest=True
        ).astype(str)
        df["charge_ratio"] = np.where(
            df["TotalCharges"] > 0, df["MonthlyCharges"] / df["TotalCharges"], 0.0
        )
        service_df = df[self.SERVICES_COLS].replace(self.BINARY_MAP)
        df["services_count"] = service_df.apply(pd.to_numeric, errors="coerce").sum(
            axis=1
        )
        df["has_internet"] = (df["InternetService"] != "No").astype(int)
        df["high_risk_contract"] = (df["Contract"] == "Month-to-month").astype(int)
        df["fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
        df["electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
        df["risk_score"] = (
            df["high_risk_contract"] * 3
            + df["fiber_optic"] * 2
            + df["electronic_check"] * 1
            + (df["tenure"] < 12).astype(int) * 2
        )
        df["charge_per_service"] = np.where(
            df["services_count"] > 0,
            df["MonthlyCharges"] / df["services_count"],
            df["MonthlyCharges"],
        )
        return df

    def _encode_and_scale(self, df, fit=True):
        target_col = self.config["data"]["target_column"]
        drop_cols = self.config["data"].get("drop_columns", [])
        y = df[target_col].copy() if target_col in df.columns else None
        X = df.drop(columns=[c for c in [target_col] + drop_cols if c in df.columns])
        binary_cols = [
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "MultipleLines",
        ]
        for col in binary_cols:
            if col in X.columns:
                X[col] = X[col].map(self.BINARY_MAP).fillna(0).astype(int)
        if "gender" in X.columns:
            X["gender"] = (X["gender"] == "Female").astype(int)
        if "Contract" in X.columns:
            X["Contract"] = (
                X["Contract"]
                .map({v: i for i, v in enumerate(self.CONTRACT_ORDER)})
                .fillna(0)
                .astype(int)
            )
        X = pd.get_dummies(
            X,
            columns=["InternetService", "PaymentMethod", "tenure_group"],
            drop_first=False,
        )
        # Add missing columns that exist in training but not in this single row
        expected_dummies = [
            "InternetService_DSL",
            "InternetService_Fiber optic",
            "InternetService_No",
            "PaymentMethod_Bank transfer (automatic)",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
            "tenure_group_developing",
            "tenure_group_established",
            "tenure_group_loyal",
            "tenure_group_new",
        ]
        for col in expected_dummies:
            if col not in X.columns:
                X[col] = False
        num_cols = [
            c
            for c in [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "charge_ratio",
                "services_count",
                "risk_score",
                "charge_per_service",
            ]
            if c in X.columns
        ]
        if fit:
            self.scaler = StandardScaler()
            X[num_cols] = self.scaler.fit_transform(X[num_cols])
            joblib.dump(self.scaler, self.artifact_dir / "scaler.joblib")
        else:
            scaler = None
            for sp in [
                str(self.artifact_dir / "scaler.joblib"),
                "artifacts/models/scaler.joblib",
                "/app/artifacts/models/scaler.joblib",
            ]:
                try:
                    scaler = joblib.load(sp)
                    break
                except Exception:
                    continue
            if scaler is None:
                raise RuntimeError("Could not load scaler")
            X[num_cols] = scaler.transform(X[num_cols])
        # Force all dummy columns to bool to match training schema
        dummy_cols = [
            "InternetService_DSL",
            "InternetService_Fiber optic",
            "InternetService_No",
            "PaymentMethod_Bank transfer (automatic)",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
            "tenure_group_developing",
            "tenure_group_established",
            "tenure_group_loyal",
            "tenure_group_new",
        ]
        for col in dummy_cols:
            if col in X.columns:
                X[col] = X[col].astype(bool)
            else:
                X[col] = False
        if y is not None:
            X[target_col] = y.values
        return X


def build_features(input_path, output_path, config, artifact_dir="artifacts/models"):
    df_clean = pd.read_parquet(input_path)
    engineer = FeatureEngineer(config=config, artifact_dir=artifact_dir)
    df_features = engineer.fit_transform(df_clean)
    engineer.save_preprocessor()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False)
    return df_features

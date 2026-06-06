"""
Model Training Module — MLflow + Optuna
────────────────────────────────────────
Trains Logistic Regression, XGBoost, and LightGBM with Optuna
hyperparameter search. Every run is tracked in MLflow.
Best model is registered in the MLflow Model Registry.
"""

import os
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import optuna
from optuna.samplers import TPESampler
import joblib
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
    }


# ─────────────────────────────────────────────────────────────────────
# Optuna objective factories
# ─────────────────────────────────────────────────────────────────────

def _lr_objective(trial, X_train, y_train, cv):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    # l1 needs liblinear/saga, l2 works with lbfgs/liblinear
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    params = {
        "C":            trial.suggest_float("C", 0.01, 100.0, log=True),
        "penalty":      penalty,
        "solver":       solver,
        "max_iter":     1000,
        "random_state": 42,
    }
    model = LogisticRegression(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    return scores.mean()

def _xgb_objective(trial, X_train, y_train, cv):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 7),
        "gamma":             trial.suggest_float("gamma", 0.0, 1.0),
        "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "use_label_encoder": False,
        "eval_metric":       "logloss",
        "random_state":      42,
        "n_jobs":            -1,
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
    return scores.mean()


def _lgb_objective(trial, X_train, y_train, cv):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 100),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "class_weight":      "balanced",
        "random_state":      42,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
    return scores.mean()


# ─────────────────────────────────────────────────────────────────────
# ModelTrainer
# ─────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Orchestrates multi-model training with:
    - SMOTE for class imbalance
    - Optuna hyperparameter optimisation
    - MLflow experiment tracking
    - Model Registry promotion
    """

    OBJECTIVES = {
        "logistic_regression": _lr_objective,
        "xgboost":             _xgb_objective,
        "lightgbm":            _lgb_objective,
    }

    def __init__(self, config: dict):
        self.config     = config
        self.target_col = config["data"]["target_column"]
        self.mlflow_cfg = config["mlflow"]
        self.hp_cfg     = config["hyperparameters"]
        self.model_cfg  = config["models"]

        # Setup MLflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.mlflow_cfg["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.mlflow_cfg["experiment_name"])

        self.best_model       = None
        self.best_model_name  = None
        self.best_metrics     = {}
        self.best_run_id      = None

    # ─────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────

    def run(self, features_path: str) -> dict:
        """
        Full training pipeline:
        1. Load features
        2. Split train/val/test
        3. SMOTE on train
        4. Optuna tune each model
        5. Evaluate on held-out test
        6. Register best model
        """
        df = pd.read_parquet(features_path)
        X  = df.drop(columns=[self.target_col])
        y  = df[self.target_col].astype(int)

        logger.info(f"Dataset: {X.shape}  |  Churn rate: {y.mean():.2%}")

        # Train / test split
        cfg      = self.config["data"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg["test_size"],
            random_state=cfg["random_state"],
            stratify=y,
        )

        # SMOTE on training set only
        if self.model_cfg.get("handle_imbalance"):
            sm = SMOTE(random_state=cfg["random_state"])
            X_train, y_train = sm.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {X_train.shape}  |  churn rate: {y_train.mean():.2%}")

        results = {}
        for model_name in self.model_cfg["candidates"]:
            logger.info(f"\n{'='*55}\nTuning: {model_name.upper()}\n{'='*55}")
            metrics, run_id = self._tune_and_log(model_name, X_train, y_train, X_test, y_test)
            results[model_name] = {"metrics": metrics, "run_id": run_id}

        # Pick best model by F1
        self._select_best(results, X_test, y_test)
        self._register_best_model()

        logger.success(
            f"\nBest Model: {self.best_model_name} | "
            f"F1={self.best_metrics['f1']} | AUC={self.best_metrics['roc_auc']}"
        )
        return results

    # ─────────────────────────────────────────────────
    # Per-model Optuna + MLflow
    # ─────────────────────────────────────────────────

    def _tune_and_log(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test:  pd.DataFrame,
        y_test:  pd.Series,
    ) -> tuple[dict, str]:

        n_trials = self.hp_cfg[model_name]["n_trials"]
        objective_fn = self.OBJECTIVES[model_name]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # ── Optuna study ──────────────────────────────
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name=f"{model_name}_study",
        )
        study.optimize(
            lambda t: objective_fn(t, X_train, y_train, cv),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        logger.info(f"Best CV F1={study.best_value:.4f}  params={best_params}")

        # ── Build final model with best params ────────
        final_model = self._build_model(model_name, best_params)
        final_model.fit(X_train, y_train)

        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.values, y_pred, y_prob)

        # ── MLflow run ────────────────────────────────
        with mlflow.start_run(run_name=f"{model_name}_optuna") as run:
            # Tags
            mlflow.set_tags({
                **self.mlflow_cfg.get("default_tags", {}),
                "model_type": model_name,
                "tuner": "optuna",
                "smote": str(self.model_cfg.get("handle_imbalance", False)),
            })
            # Hyperparameters
            mlflow.log_params(best_params)
            mlflow.log_param("n_optuna_trials", n_trials)
            # Metrics
            mlflow.log_metrics(metrics)
            mlflow.log_metric("optuna_best_cv_f1", study.best_value)

            # Optuna importance plot as artifact
            
            try:
                import optuna.visualization as vis
                fig = vis.plot_param_importances(study)
                plot_path = f"/tmp/{model_name}_param_importance.html"
                fig.write_html(plot_path)
                mlflow.log_artifact(plot_path, "optuna_plots")
            except Exception:
                pass

            # Classification report
            try:
                report = classification_report(y_test, y_pred, output_dict=True)
                report_path = f"/tmp/{model_name}_report.json"
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
                mlflow.log_artifact(report_path, "evaluation")
            except Exception:
                pass

            # Log model with signature
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, y_pred)

            try:
                if model_name == "xgboost":
                    mlflow.xgboost.log_model(final_model, "model", signature=signature)
                elif model_name == "lightgbm":
                    mlflow.lightgbm.log_model(final_model, "model", signature=signature)
                else:
                    mlflow.sklearn.log_model(final_model, "model", signature=signature)
            except Exception as e:
                logger.warning(f"Could not log model artifact: {e}")
                mlflow.sklearn.log_model(final_model, "model", signature=signature)

            run_id = run.info.run_id

        logger.info(f"{model_name} → {metrics}  run_id={run_id}")
        return metrics, run_id

    # ─────────────────────────────────────────────────
    # Model selection & registry
    # ─────────────────────────────────────────────────

    def _select_best(self, results: dict, X_test, y_test) -> None:
        best_f1 = -1
        for name, result in results.items():
            f1 = result["metrics"]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name
                self.best_metrics    = result["metrics"]
                self.best_run_id     = result["run_id"]

        logger.info(f"Selected: {self.best_model_name} (F1={best_f1:.4f})")

    def _register_best_model(self) -> None:
        model_uri  = f"runs:/{self.best_run_id}/model"
        model_name = self.mlflow_cfg["model_name"]

        registered = mlflow.register_model(model_uri=model_uri, name=model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage="Staging",
        )
        logger.success(
            f"Registered '{model_name}' v{registered.version} → Staging"
        )

    # ─────────────────────────────────────────────────
    # Model factories
    # ─────────────────────────────────────────────────

    @staticmethod
    def _build_model(name: str, params: dict):
        if name == "logistic_regression":
            # Remove max_iter and random_state from params if already set
            clean_params = {k: v for k, v in params.items() 
                          if k not in ["max_iter", "random_state"]}
            return LogisticRegression(**clean_params, max_iter=1000, random_state=42)
        elif name == "xgboost":
            return xgb.XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        elif name == "lightgbm":
            return lgb.LGBMClassifier(
                **params,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            raise ValueError(f"Unknown model: {name}")


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

def main():
    import yaml
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    features_path = config["paths"]["features_data"]
    trainer = ModelTrainer(config)
    results = trainer.run(features_path)

    print("\n===== EXPERIMENT SUMMARY =====")
    for model, data in results.items():
        print(f"\n{model.upper()}")
        for metric, val in data["metrics"].items():
            print(f"  {metric:12s}: {val:.4f}")


if __name__ == "__main__":
    main()

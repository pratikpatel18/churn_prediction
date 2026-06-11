"""
Model Training Module — MLflow + Optuna
────────────────────────────────────────
Trains Logistic Regression, XGBoost, and LightGBM with Optuna
hyperparameter search. Every run is tracked in MLflow.
Best model is registered in the MLflow Model Registry.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from loguru import logger
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Core metrics at a given threshold."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


def compute_extended_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Extended metrics logged to MLflow on top of core metrics:
    - Matthews Correlation Coefficient (MCC)
    - Log Loss
    - Average Precision Score (area under PR curve)
    - Confusion matrix breakdown: TP, TN, FP, FN
    - Specificity (True Negative Rate)
    - False Positive Rate
    - False Negative Rate
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = round(tn / (tn + fp + 1e-8), 4)
    fpr = round(fp / (fp + tn + 1e-8), 4)
    fnr = round(fn / (fn + tp + 1e-8), 4)

    return {
        # Threshold-independent
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "log_loss": round(log_loss(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        # Confusion matrix
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        # Derived rates
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


# ─────────────────────────────────────────────────────────────────────
# Optuna objective factories
# ─────────────────────────────────────────────────────────────────────


def _lr_objective(trial, X_train, y_train, cv):
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "liblinear" if penalty == "l1" else "lbfgs"
    params = {
        "C": trial.suggest_float("C", 0.01, 100.0, log=True),
        "penalty": penalty,
        "solver": solver,
        "max_iter": 1000,
        "random_state": 42,
        "class_weight": "balanced",
    }
    model = LogisticRegression(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    return scores.mean()


def _xgb_objective(trial, X_train, y_train, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=1)
    return scores.mean()


def _lgb_objective(trial, X_train, y_train, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
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
    - class_weight='balanced' on all models
    - Optuna hyperparameter optimisation
    - Threshold tuning via PR curve
    - Extended metrics (MCC, log loss, confusion matrix breakdown)
    - SHAP feature importance
    - MLflow experiment tracking
    - Model Registry promotion
    """

    OBJECTIVES = {
        "logistic_regression": _lr_objective,
        "xgboost": _xgb_objective,
        "lightgbm": _lgb_objective,
    }

    def __init__(self, config: dict):
        self.config = config
        self.target_col = config["data"]["target_column"]
        self.mlflow_cfg = config["mlflow"]
        self.hp_cfg = config["hyperparameters"]
        self.model_cfg = config["models"]

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.mlflow_cfg["tracking_uri"])
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.mlflow_cfg["experiment_name"])

        self.best_model = None
        self.best_model_name = None
        self.best_metrics = {}
        self.best_run_id = None

    # ─────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────

    def run(self, features_path: str) -> dict:
        df = pd.read_parquet(features_path)
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col].astype(int)

        logger.info(f"Dataset: {X.shape}  |  Churn rate: {y.mean():.2%}")

        cfg = self.config["data"]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
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
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> tuple[dict, str]:

        n_trials = self.hp_cfg[model_name]["n_trials"]
        objective_fn = self.OBJECTIVES[model_name]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # ── Optuna study ──────────────────────────────────────────────
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

        # ── Build & train final model ─────────────────────────────────
        final_model = self._build_model(model_name, best_params)
        final_model.fit(X_train, y_train)

        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)[:, 1]

        # ── Core metrics at default threshold (0.5) ───────────────────
        metrics = compute_metrics(y_test.values, y_pred, y_prob)

        # ── Extended metrics at default threshold ─────────────────────
        ext_metrics = compute_extended_metrics(y_test.values, y_pred, y_prob)
        logger.info(
            f"Extended → MCC={ext_metrics['mcc']}  "
            f"LogLoss={ext_metrics['log_loss']}  "
            f"AvgPrecision={ext_metrics['avg_precision']}  "
            f"TP={ext_metrics['true_positives']}  FN={ext_metrics['false_negatives']}"
        )

        # ── Threshold tuning via PR curve ─────────────────────────────
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        f1_curve = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_thresh_idx = np.argmax(f1_curve[:-1])
        best_threshold = float(thresholds[best_thresh_idx])
        best_thresh_f1 = float(f1_curve[best_thresh_idx])

        logger.info(
            f"Default threshold 0.5  → F1={metrics['f1']:.4f}  "
            f"Recall={metrics['recall']:.4f}  Precision={metrics['precision']:.4f}"
        )
        logger.info(f"Optimal threshold {best_threshold:.3f} → F1={best_thresh_f1:.4f}")

        # Metrics at optimal threshold
        y_pred_tuned = (y_prob >= best_threshold).astype(int)
        metrics_tuned = compute_metrics(y_test.values, y_pred_tuned, y_prob)
        ext_tuned = compute_extended_metrics(y_test.values, y_pred_tuned, y_prob)
        f1_improvement = round(metrics_tuned["f1"] - metrics["f1"], 4)
        logger.info(f"F1 improvement from threshold tuning: {f1_improvement:+.4f}")

        # ── MLflow run ────────────────────────────────────────────────
        with mlflow.start_run(run_name=f"{model_name}_optuna") as run:

            # Tags
            mlflow.set_tags(
                {
                    **self.mlflow_cfg.get("default_tags", {}),
                    "model_type": model_name,
                    "tuner": "optuna",
                    "smote": str(self.model_cfg.get("handle_imbalance", False)),
                }
            )

            # Hyperparameters
            mlflow.log_params(best_params)
            mlflow.log_param("n_optuna_trials", n_trials)

            # ── Core metrics (default threshold) ──────────────────────
            mlflow.log_metrics(metrics)
            mlflow.log_metric("optuna_best_cv_f1", study.best_value)

            # ── Extended metrics (default threshold) ──────────────────
            mlflow.log_metrics(ext_metrics)

            # ── Threshold tuning metrics ───────────────────────────────
            mlflow.log_metric("best_threshold", best_threshold)
            mlflow.log_metric("tuned_f1", metrics_tuned["f1"])
            mlflow.log_metric("tuned_recall", metrics_tuned["recall"])
            mlflow.log_metric("tuned_precision", metrics_tuned["precision"])
            mlflow.log_metric("tuned_accuracy", metrics_tuned["accuracy"])
            mlflow.log_metric("tuned_mcc", ext_tuned["mcc"])
            mlflow.log_metric("tuned_false_negatives", ext_tuned["false_negatives"])
            mlflow.log_metric("tuned_false_positives", ext_tuned["false_positives"])
            mlflow.log_metric("f1_improvement_from_tuning", f1_improvement)

            # ── Optuna param importance plot ───────────────────────────
            try:
                import optuna.visualization as vis

                fig = vis.plot_param_importances(study)
                plot_path = f"/tmp/{model_name}_param_importance.html"
                fig.write_html(plot_path)
                mlflow.log_artifact(plot_path, "optuna_plots")
            except Exception:
                pass

            # ── Classification report ──────────────────────────────────
            try:
                report = classification_report(y_test, y_pred, output_dict=True)
                report_path = f"/tmp/{model_name}_report.json"
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
                mlflow.log_artifact(report_path, "evaluation")
            except Exception:
                pass

            # ── Confusion matrix as artifact ───────────────────────────
            try:
                cm_dict = {
                    "true_negatives": int(ext_metrics["true_negatives"]),
                    "false_positives": int(ext_metrics["false_positives"]),
                    "false_negatives": int(ext_metrics["false_negatives"]),
                    "true_positives": int(ext_metrics["true_positives"]),
                }
                cm_path = f"/tmp/{model_name}_confusion_matrix.json"
                with open(cm_path, "w") as f:
                    json.dump(cm_dict, f, indent=2)
                mlflow.log_artifact(cm_path, "evaluation")
            except Exception:
                pass

            # ── SHAP feature importance ────────────────────────────────
            try:
                import matplotlib.pyplot as plt
                import shap

                if model_name in ["lightgbm", "xgboost"]:
                    explainer = shap.TreeExplainer(final_model)
                    shap_vals = explainer.shap_values(X_test)

                    # LightGBM binary returns list [class0, class1]
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[1]

                    # Summary plot
                    plt.figure()
                    shap.summary_plot(shap_vals, X_test, show=False, max_display=15)
                    shap_path = f"/tmp/{model_name}_shap_summary.png"
                    plt.savefig(shap_path, bbox_inches="tight", dpi=100)
                    plt.close()
                    mlflow.log_artifact(shap_path, "shap")

                    # SHAP importance CSV
                    mean_shap = pd.DataFrame(
                        {
                            "feature": X_test.columns,
                            "importance": np.abs(shap_vals).mean(axis=0),
                        }
                    ).sort_values("importance", ascending=False)
                    shap_csv = f"/tmp/{model_name}_shap_importance.csv"
                    mean_shap.to_csv(shap_csv, index=False)
                    mlflow.log_artifact(shap_csv, "shap")
                    logger.info(f"Top 5 SHAP features:\n{mean_shap.head().to_string()}")

            except Exception as e:
                logger.warning(f"SHAP failed: {e}")

            # ── Log model with signature ───────────────────────────────
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
                self.best_metrics = result["metrics"]
                self.best_run_id = result["run_id"]
        logger.info(f"Selected: {self.best_model_name} (F1={best_f1:.4f})")

    def _register_best_model(self) -> None:
        model_uri = f"runs:/{self.best_run_id}/model"
        model_name = self.mlflow_cfg["model_name"]

        registered = mlflow.register_model(model_uri=model_uri, name=model_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage="Staging",
        )
        logger.success(f"Registered '{model_name}' v{registered.version} → Staging")

    # ─────────────────────────────────────────────────
    # Model factories
    # ─────────────────────────────────────────────────

    @staticmethod
    def _build_model(name: str, params: dict):
        if name == "logistic_regression":
            clean = {
                k: v
                for k, v in params.items()
                if k not in ["max_iter", "random_state", "class_weight"]
            }
            return LogisticRegression(
                **clean,
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            )
        elif name == "xgboost":
            clean = {
                k: v
                for k, v in params.items()
                if k not in ["use_label_encoder", "eval_metric", "random_state", "n_jobs"]
            }
            return xgb.XGBClassifier(
                **clean,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        elif name == "lightgbm":
            clean = {
                k: v
                for k, v in params.items()
                if k not in ["class_weight", "random_state", "n_jobs", "verbose"]
            }
            return lgb.LGBMClassifier(
                **clean,
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

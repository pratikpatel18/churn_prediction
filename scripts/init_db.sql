-- ─────────────────────────────────────────────────────────────────
-- PostgreSQL Initialization Script
-- Runs automatically when the postgres container starts for the first time
-- Creates: airflow_db, mlflow_db (churn_db already created by POSTGRES_DB)
-- ─────────────────────────────────────────────────────────────────

-- Create Airflow database
CREATE DATABASE airflow_db;

-- Create MLflow database
CREATE DATABASE mlflow_db;

-- Grant access to churn_user on all databases
GRANT ALL PRIVILEGES ON DATABASE airflow_db TO churn_user;
GRANT ALL PRIVILEGES ON DATABASE mlflow_db  TO churn_user;
GRANT ALL PRIVILEGES ON DATABASE churn_db   TO churn_user;

-- ── Connect to churn_db and create app tables ─────────────────────
\c churn_db;

-- Prediction log
CREATE TABLE IF NOT EXISTS prediction_log (
    id                SERIAL PRIMARY KEY,
    request_id        VARCHAR(50)  NOT NULL,
    customer_id       VARCHAR(50),
    churn_probability FLOAT        NOT NULL,
    churn_prediction  INT          NOT NULL,
    risk_label        VARCHAR(20),
    model_version     VARCHAR(50),
    created_at        TIMESTAMP    DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pred_created ON prediction_log(created_at);
CREATE INDEX IF NOT EXISTS idx_pred_risk    ON prediction_log(risk_label);

-- Model performance history
CREATE TABLE IF NOT EXISTS model_performance_log (
    id                SERIAL PRIMARY KEY,
    check_date        DATE,
    model_version     VARCHAR(50),
    accuracy          FLOAT,
    f1_score          FLOAT,
    roc_auc           FLOAT,
    triggered_retrain BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMP DEFAULT NOW()
);

-- Drift report log
CREATE TABLE IF NOT EXISTS drift_report_log (
    id                  SERIAL PRIMARY KEY,
    report_date         DATE,
    dataset_drift       BOOLEAN,
    n_drifted_features  INT,
    share_drifted       FLOAT,
    all_tests_passed    BOOLEAN,
    created_at          TIMESTAMP DEFAULT NOW()
);

\echo '✅ churn_db tables created'

-- ── Connect to airflow_db ──────────────────────────────────────────
\c airflow_db;
GRANT ALL PRIVILEGES ON SCHEMA public TO churn_user;

-- ── Connect to mlflow_db ───────────────────────────────────────────
\c mlflow_db;
GRANT ALL PRIVILEGES ON SCHEMA public TO churn_user;

\echo '✅ Database initialization complete'

"""
Streamlit Monitoring Dashboard
───────────────────────────────
Live view of:
  • Model performance metrics
  • Data drift alerts (Evidently)
  • Prediction volume & churn rate trends
  • Feature distributions (reference vs current)
  • Experiment comparison (MLflow runs)

Run:  streamlit run dashboard/streamlit_app.py
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parents[1]))

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Prediction | MLOps Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# Load config & data
# ─────────────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data(ttl=60)
def load_drift_report(path: str) -> dict | None:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=120)
def load_features(path: str) -> pd.DataFrame | None:
    if Path(path).exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=120)
def load_reference(path: str) -> pd.DataFrame | None:
    if Path(path).exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=300)
def load_mlflow_runs(tracking_uri: str, experiment_name: str) -> pd.DataFrame | None:
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.f1 DESC"],
            max_results=30,
        )
        return runs
    except Exception:
        return None


def _color_alert(level: str) -> str:
    return {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🟢"}.get(level, "⚪")


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────

config = load_config()

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("Churn MLOps")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Data Drift", "🏆 Experiments", "🔮 Live Predict"],
    )

    st.markdown("---")
    st.caption(f"MLflow: {config['mlflow']['tracking_uri']}")
    st.caption(f"Model: {config['mlflow']['model_name']}")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────

REPORT_PATH = "artifacts/reports/latest_drift_report.json"
FEATURES_PATH = config["paths"]["features_data"]
REFERENCE_PATH = config["paths"]["reference_data"]

drift_report = load_drift_report(REPORT_PATH)
features_df = load_features(FEATURES_PATH)
reference_df = load_reference(REFERENCE_PATH)
mlflow_runs = load_mlflow_runs(
    config["mlflow"]["tracking_uri"],
    config["mlflow"]["experiment_name"],
)


# ─────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Customer Churn Prediction — MLOps Dashboard")
    st.markdown("Real-time model health, data quality, and business KPIs.")
    st.markdown("---")

    # ── Top KPI cards ─────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    # Derive metrics from available data
    if features_df is not None:
        target = config["data"]["target_column"]
        churn_rate = features_df[target].mean() if target in features_df.columns else None
        n_customers = len(features_df)
    else:
        churn_rate, n_customers = None, None

    # Get best MLflow metrics
    best_f1, best_auc = None, None
    if mlflow_runs is not None and not mlflow_runs.empty:
        if "metrics.f1" in mlflow_runs.columns:
            best_f1 = mlflow_runs["metrics.f1"].max()
            best_auc = (
                mlflow_runs["metrics.roc_auc"].max()
                if "metrics.roc_auc" in mlflow_runs.columns
                else None
            )

    col1.metric("👥 Total Customers", f"{n_customers:,}" if n_customers else "N/A")
    col2.metric("📉 Churn Rate", f"{churn_rate:.1%}" if churn_rate else "N/A")
    col3.metric("🎯 Best F1 Score", f"{best_f1:.3f}" if best_f1 else "N/A")
    col4.metric("📈 Best ROC-AUC", f"{best_auc:.3f}" if best_auc else "N/A")

    # Drift status
    if drift_report:
        drifted = drift_report.get("drift_summary", {}).get("dataset_drift", False)
        col5.metric("⚡ Data Drift", "YES 🔴" if drifted else "NO 🟢")
    else:
        col5.metric("⚡ Data Drift", "No report")

    st.markdown("---")

    # ── Alerts ────────────────────────────────────────────────────
    if drift_report and drift_report.get("alerts"):
        st.subheader("🚨 Active Alerts")
        for alert in drift_report["alerts"]:
            icon = _color_alert(alert["level"])
            st.warning(f"{icon} **{alert['level']}** — {alert['msg']}")
    else:
        st.success("✅ No active alerts — model and data are healthy")

    st.markdown("---")

    # ── Feature distributions ─────────────────────────────────────
    if features_df is not None:
        st.subheader("📦 Feature Distributions")
        num_cols = features_df.select_dtypes(include="number").columns.tolist()
        key_cols = [
            c for c in ["tenure", "MonthlyCharges", "TotalCharges", "risk_score"] if c in num_cols
        ][:4]

        if key_cols:
            fig = make_subplots(rows=1, cols=len(key_cols), subplot_titles=key_cols)
            for i, col in enumerate(key_cols, 1):
                fig.add_trace(
                    go.Histogram(
                        x=features_df[col],
                        name=col,
                        marker_color="#6366f1",
                        opacity=0.75,
                    ),
                    row=1,
                    col=i,
                )
            fig.update_layout(height=300, showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        # Churn by contract type
        if "Contract" in features_df.columns and "Churn" in features_df.columns:
            st.subheader("📊 Churn Rate by Contract Type")
            contract_map = {0: "Month-to-month", 1: "One year", 2: "Two year"}
            plot_df = features_df.copy()
            plot_df["Contract_label"] = plot_df["Contract"].map(contract_map)
            grp = plot_df.groupby("Contract_label")["Churn"].mean().reset_index()
            grp.columns = ["Contract", "Churn Rate"]
            fig2 = px.bar(
                grp,
                x="Contract",
                y="Churn Rate",
                color="Churn Rate",
                color_continuous_scale="RdYlGn_r",
                text_auto=".1%",
            )
            fig2.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: DATA DRIFT
# ─────────────────────────────────────────────────────────────────────

elif page == "🔍 Data Drift":
    st.title("🔍 Data Drift Detection — Evidently AI")
    st.markdown("Comparing **reference** (training) data vs **current** production data.")
    st.markdown("---")

    if drift_report is None:
        st.info(
            "No drift report found. Run `python scripts/run_monitoring.py` to generate one.",
            icon="ℹ️",
        )
    else:
        ds = drift_report.get("drift_summary", {})
        ts = drift_report.get("timestamp", "N/A")

        st.caption(f"Report generated: {ts}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset Drift", "YES 🔴" if ds.get("dataset_drift") else "NO 🟢")
        col2.metric(
            "Drifted Features",
            f"{ds.get('n_drifted_features', 0)} / {ds.get('n_features', 0)}",
        )
        col3.metric("Share Drifted", f"{ds.get('share_drifted', 0):.1%}")

        st.markdown("---")

        # Test suite results
        ts_results = drift_report.get("test_results", {})
        if ts_results:
            st.subheader("🧪 Test Suite Results")
            all_ok = ts_results.get("all_passed", False)
            n_fail = ts_results.get("n_failed", 0)
            n_tot = ts_results.get("n_tests", 0)

            if all_ok:
                st.success(f"All {n_tot} tests passed ✓")
            else:
                st.error(f"{n_fail}/{n_tot} tests FAILED")

            details = ts_results.get("test_details", [])
            if details:
                test_df = pd.DataFrame(details)
                st.dataframe(
                    test_df.style.applymap(
                        lambda v: ("color: #22c55e" if v == "SUCCESS" else "color: #ef4444"),
                        subset=["status"],
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("---")

        # Reference vs Current distributions
        if reference_df is not None and features_df is not None:
            st.subheader("📊 Reference vs Current Distributions")
            num_cols = [
                c
                for c in reference_df.select_dtypes(include="number").columns
                if c in features_df.columns and c != "Churn"
            ][:6]

            if num_cols:
                selected = st.selectbox("Select feature", num_cols)
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=reference_df[selected],
                        name="Reference",
                        opacity=0.6,
                        marker_color="#6366f1",
                        histnorm="probability density",
                    )
                )
                fig.add_trace(
                    go.Histogram(
                        x=features_df[selected],
                        name="Current",
                        opacity=0.6,
                        marker_color="#f59e0b",
                        histnorm="probability density",
                    )
                )
                fig.update_layout(
                    barmode="overlay",
                    height=350,
                    title=f"Distribution: {selected}",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Performance summary
        perf = drift_report.get("perf_summary", {})
        if perf:
            st.subheader("🎯 Model Performance on Current Data")
            metrics_df = pd.DataFrame([perf]).T.reset_index()
            metrics_df.columns = ["Metric", "Value"]
            metrics_df["Value"] = metrics_df["Value"].apply(
                lambda x: f"{x:.4f}" if x is not None else "N/A"
            )
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Full HTML report link
        html_reports = list(Path(config["paths"]["reports"]).glob("drift_*.html"))
        if html_reports:
            latest = sorted(html_reports)[-1]
            st.info(f"📄 Full HTML report: `{latest}` — open in browser for interactive view")


# ─────────────────────────────────────────────────────────────────────
# PAGE: EXPERIMENTS
# ─────────────────────────────────────────────────────────────────────

elif page == "🏆 Experiments":
    st.title("🏆 MLflow Experiment Tracker")
    st.markdown("Compare all training runs across models, metrics, and hyperparameters.")
    st.markdown("---")

    if mlflow_runs is None or mlflow_runs.empty:
        st.info(
            "No MLflow runs found. Run `python src/models/train.py` to start training.",
            icon="ℹ️",
        )
    else:
        # Filter metric columns
        metric_cols = [c for c in mlflow_runs.columns if c.startswith("metrics.")]
        param_cols = [c for c in mlflow_runs.columns if c.startswith("params.")]
        tag_cols = [c for c in mlflow_runs.columns if c.startswith("tags.")]

        display_cols = (
            ["run_id", "tags.model_type"]
            + [
                c
                for c in metric_cols
                if any(m in c for m in ["f1", "accuracy", "roc_auc", "recall"])
            ]
            + ["start_time"]
        )
        display_cols = [c for c in display_cols if c in mlflow_runs.columns]

        st.subheader("📋 All Runs")
        disp = mlflow_runs[display_cols].copy()
        disp.columns = [
            c.replace("metrics.", "").replace("tags.", "").replace("params.", "")
            for c in display_cols
        ]
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Metric comparison chart
        if "metrics.f1" in mlflow_runs.columns and "tags.model_type" in mlflow_runs.columns:
            st.subheader("📊 Model Comparison")
            metric_choice = st.selectbox(
                "Metric",
                [c.replace("metrics.", "") for c in metric_cols],
                index=0,
            )
            fig = px.bar(
                mlflow_runs.sort_values(f"metrics.{metric_choice}", ascending=False),
                x="tags.model_type",
                y=f"metrics.{metric_choice}",
                color=f"metrics.{metric_choice}",
                color_continuous_scale="Viridis",
                text_auto=".4f",
                title=f"{metric_choice} by Model Type",
            )
            fig.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            # Scatter: F1 vs AUC
            if "metrics.roc_auc" in mlflow_runs.columns:
                st.subheader("🎯 F1 vs ROC-AUC")
                fig2 = px.scatter(
                    mlflow_runs,
                    x="metrics.f1",
                    y="metrics.roc_auc",
                    color=("tags.model_type" if "tags.model_type" in mlflow_runs.columns else None),
                    size_max=15,
                    hover_data=["run_id"],
                    title="F1 vs ROC-AUC (all runs)",
                )
                fig2.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: LIVE PREDICT
# ─────────────────────────────────────────────────────────────────────

elif page == "🔮 Live Predict":
    st.title("🔮 Live Churn Prediction")
    st.markdown("Enter customer details to get a real-time churn prediction via the API.")
    st.markdown("---")

    api_url = os.getenv("API_BASE_URL", f"http://localhost:{config['api']['port']}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Customer Details")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    with col2:
        st.subheader("🌐 Services")
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    with col3:
        st.subheader("💳 Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.slider("Monthly Charges ($)", 10.0, 120.0, 65.0, 0.5)
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure)
        )

    st.markdown("---")

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        payload = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        try:
            import requests

            with st.spinner("Calling prediction API …"):
                resp = requests.post(f"{api_url}/predict", json=payload, timeout=10)

            if resp.status_code == 200:
                result = resp.json()
                prob = result["churn_probability"]
                pred = result["churn_prediction"]
                risk = result["risk_label"]

                c1, c2, c3 = st.columns(3)
                c1.metric("Churn Probability", f"{prob:.1%}")
                c2.metric("Prediction", "Will Churn 🔴" if pred == 1 else "Will Stay 🟢")
                c3.metric("Risk Level", risk)

                # Gauge chart
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=prob * 100,
                        title={"text": "Churn Risk Score"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#ef4444" if prob > 0.5 else "#22c55e"},
                            "steps": [
                                {"range": [0, 30], "color": "#dcfce7"},
                                {"range": [30, 50], "color": "#fef9c3"},
                                {"range": [50, 75], "color": "#fee2e2"},
                                {"range": [75, 100], "color": "#fca5a5"},
                            ],
                            "threshold": {
                                "value": 50,
                                "line": {"color": "black", "width": 3},
                            },
                        },
                    )
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error(f"API error {resp.status_code}: {resp.text}")

        except Exception as e:
            st.error(f"Could not reach API at {api_url}. Is the FastAPI server running?\n\n`{e}`")
            st.code(f"uvicorn src.api.main:app --host 0.0.0.0 --port {config['api']['port']}")

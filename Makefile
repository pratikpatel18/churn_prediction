# ─────────────────────────────────────────────────────────────────────
# Makefile — End-to-End Churn Prediction
# Usage:  make <target>
# ─────────────────────────────────────────────────────────────────────

.PHONY: help setup install env lint format test pipeline train monitor \
        api dashboard mlflow docker-up docker-down docker-build \
        dvc-init dvc-push clean

PYTHON      = python3
PIP         = pip3
CONFIG      = configs/config.yaml
DATA_PATH   = data/raw/churn_data.csv


# ── Default ────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════╗"
	@echo "  ║   Customer Churn Prediction — Developer Commands     ║"
	@echo "  ╚══════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  SETUP"
	@echo "    make setup         Full first-time setup"
	@echo "    make install       Install Python dependencies"
	@echo "    make env           Copy .env.example → .env"
	@echo ""
	@echo "  CODE QUALITY"
	@echo "    make lint          Run flake8 linter"
	@echo "    make format        Auto-format with black + isort"
	@echo "    make test          Run full pytest suite with coverage"
	@echo ""
	@echo "  PIPELINE"
	@echo "    make pipeline      Full pipeline (ingest→features→train→monitor)"
	@echo "    make train         Model training only (MLflow + Optuna)"
	@echo "    make monitor       Drift monitoring report"
	@echo "    make quality-gate  Check model meets performance thresholds"
	@echo ""
	@echo "  SERVICES"
	@echo "    make mlflow        Start MLflow tracking server"
	@echo "    make api           Start FastAPI prediction server"
	@echo "    make dashboard     Start Streamlit monitoring dashboard"
	@echo ""
	@echo "  DOCKER"
	@echo "    make docker-build  Build all Docker images"
	@echo "    make docker-up     Start full Docker Compose stack"
	@echo "    make docker-down   Stop Docker Compose stack"
	@echo ""
	@echo "  DVC"
	@echo "    make dvc-init      Initialize DVC in this repo"
	@echo "    make dvc-push      Push data to DVC remote"
	@echo "    make dvc-repro     Re-run full DVC pipeline"
	@echo ""
	@echo "  CLEANUP"
	@echo "    make clean         Remove cache, temp files"
	@echo ""


# ── Setup ──────────────────────────────────────────────────────────
setup: env install dvc-init create-dirs
	@echo "✅ Setup complete! Next: add your dataset to data/raw/churn_data.csv"
	@echo "   Then run:  make pipeline"

env:
	@if [ ! -f .env ]; then cp .env.example .env; echo "✅ .env created — fill in your values"; \
	else echo "ℹ️  .env already exists"; fi

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "✅ Dependencies installed"

create-dirs:
	mkdir -p data/raw data/processed data/features data/reference data/raw_parquet
	mkdir -p artifacts/models artifacts/reports
	mkdir -p great_expectations logs
	@echo "✅ Directories created"


# ── Code Quality ──────────────────────────────────────────────────
lint:
	flake8 src/ dags/ dashboard/ tests/ --max-line-length=100 --ignore=E203,W503
	@echo "✅ Lint passed"

format:
	black src/ dags/ dashboard/ tests/ scripts/
	isort src/ dags/ dashboard/ tests/ scripts/
	@echo "✅ Formatting applied"

test:
	pytest tests/ -v \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:artifacts/reports/coverage \
		--tb=short
	@echo "✅ Tests complete — coverage at artifacts/reports/coverage/index.html"

test-fast:
	pytest tests/ -v -x --tb=short -q


# ── Pipeline ──────────────────────────────────────────────────────
pipeline:
	$(PYTHON) scripts/run_pipeline.py --data-path $(DATA_PATH)

pipeline-no-train:
	$(PYTHON) scripts/run_pipeline.py --skip-training

train:
	$(PYTHON) src/models/train.py

monitor:
	$(PYTHON) scripts/run_monitoring.py --simulate-predictions

quality-gate:
	$(PYTHON) scripts/check_model_quality.py

dvc-repro:
	dvc repro


# ── Services (run in separate terminals) ─────────────────────────
mlflow:
	@echo "🚀 Starting MLflow at http://localhost:5000 ..."
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlflow-artifacts \
		--host 0.0.0.0 \
		--port 5000

api:
	@echo "🚀 Starting FastAPI at http://localhost:8000 ..."
	@echo "   Docs: http://localhost:8000/docs"
	uvicorn src.api.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

dashboard:
	@echo "🚀 Starting Streamlit at http://localhost:8501 ..."
	streamlit run dashboard/streamlit_app.py \
		--server.port 8501 \
		--server.address 0.0.0.0


# ── Docker ────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile.api       -t churn-api:latest .
	docker build -f docker/Dockerfile.streamlit -t churn-dashboard:latest .
	@echo "✅ Docker images built"

docker-up:
	docker compose up -d
	@echo "✅ Stack running:"
	@echo "   Airflow   → http://localhost:8080  (admin/admin)"
	@echo "   MLflow    → http://localhost:5000"
	@echo "   API       → http://localhost:8000/docs"
	@echo "   Dashboard → http://localhost:8501"

docker-down:
	docker compose down
	@echo "✅ Stack stopped"

docker-logs:
	docker compose logs -f --tail=100

docker-restart:
	docker compose restart


# ── DVC ──────────────────────────────────────────────────────────
dvc-init:
	@if [ ! -d .dvc ]; then \
		dvc init; \
		echo "✅ DVC initialized"; \
	else \
		echo "ℹ️  DVC already initialized"; \
	fi

dvc-push:
	dvc add data/features/churn_features.parquet
	dvc push
	@echo "✅ DVC push complete"

dvc-pull:
	dvc pull
	@echo "✅ DVC pull complete"


# ── Cleanup ────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"        -delete 2>/dev/null || true
	find . -type f -name "*.pyo"        -delete 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/ model_eval_result.txt
	@echo "✅ Cleaned"

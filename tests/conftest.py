"""
conftest.py — Shared pytest fixtures and configuration.
Automatically loaded by pytest before running any tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parents[1]))

# Use SQLite for MLflow during tests (no server needed)
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow_test.db")
os.environ.setdefault("CONFIG_PATH", "configs/config.yaml")


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (skip with -m 'not slow')")
    config.addinivalue_line("markers", "integration: marks integration tests (need live services)")
    config.addinivalue_line("markers", "api: marks API endpoint tests")


@pytest.fixture(scope="session", autouse=True)
def create_test_dirs():
    """Create required directories before any tests run."""
    dirs = [
        "data/raw", "data/processed", "data/features",
        "data/reference", "data/raw_parquet",
        "artifacts/models", "artifacts/reports",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup test SQLite DB
    if Path("mlflow_test.db").exists():
        Path("mlflow_test.db").unlink()

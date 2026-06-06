from setuptools import setup, find_packages

setup(
    name="churn_prediction",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="End-to-End Customer Churn Prediction MLOps Pipeline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.2.0",
        "scikit-learn>=1.5.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.3.0",
        "mlflow>=2.13.0",
        "fastapi>=0.111.0",
        "evidently>=0.4.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
        "airflow": ["apache-airflow>=2.9.0"],
    },
    entry_points={
        "console_scripts": [
            "churn-train=src.models.train:main",
            "churn-serve=src.api.main:start_server",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)

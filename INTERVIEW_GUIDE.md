# 🎯 Interview Guide — How to Present This Project

> Use this doc to prepare your talking points.
> Every section maps to a real interview question you WILL be asked.

---

## ⚡ 60-Second Elevator Pitch

> *"I built an end-to-end MLOps pipeline for customer churn prediction that goes well beyond
> a Jupyter notebook. It ingests data through an Airflow DAG, validates it with Great
> Expectations, engineers features using domain knowledge from the Telco dataset, then
> runs three models through Optuna hyperparameter search — all tracked in MLflow.
> The best model gets registered and deployed via FastAPI.
> The killer feature is the automated monitoring: Evidently AI checks for data drift weekly,
> and if model performance drops below threshold, Airflow automatically triggers retraining
> and promotes the new model to production — zero manual intervention."*

---

## 💬 Q&A — Questions You Will Be Asked

---

### Q: "Walk me through your project architecture."

**Answer framework — follow this exact order:**

1. **Data Layer** → "Raw CSV ingested via Airflow, validated with Great Expectations (schema, nulls, value ranges), cleaned and saved as Parquet."
2. **Feature Layer** → "DVC versions the datasets. Pandas builds domain features — tenure buckets, risk score, charge ratio — then scales and encodes."
3. **Model Layer** → "Three models compared: Logistic Regression as baseline, XGBoost and LightGBM as boosters. Optuna runs 50 trials each, tracking every run in MLflow."
4. **Serving Layer** → "Best model registered in MLflow Model Registry. FastAPI wraps it — single predict, batch predict, health check endpoints."
5. **Monitoring Layer** → "Evidently AI compares reference vs current data weekly. Streamlit dashboard shows live metrics. If F1 drops below 0.75, Airflow auto-retrains."

**Pro tip:** Pull up the Streamlit dashboard during this answer. Seeing a live dashboard shuts down follow-up questions.

---

### Q: "Why did you use DVC?"

> *"DVC solves the reproducibility problem that kills most ML projects in teams.
> Without it, you have models but you don't know which dataset version trained them.
> DVC treats data like Git treats code — every experiment is tied to an exact data snapshot.
> If a colleague joins 6 months later, they run `dvc pull` and `dvc repro` and get
> the exact same model I have. That's real team collaboration."*

---

### Q: "Why Optuna instead of GridSearch?"

> *"GridSearch is exhaustive — it checks every combination, which is O(n^k) for k params.
> Optuna uses Tree-structured Parzen Estimator (TPE), a Bayesian approach that learns
> from previous trials. It figures out which hyperparameter regions are promising and
> focuses there. In practice it finds better results in 50 trials than GridSearch
> does in 500. It also supports pruning — stopping bad trials early — which is huge
> for tree models with hundreds of estimators."*

---

### Q: "How does your monitoring work?"

> *"Evidently AI runs a statistical test — by default Population Stability Index or
> Jensen-Shannon divergence — comparing each feature's distribution in the reference
> dataset (what the model trained on) versus what's arriving in production.
> If more than 30% of features drift significantly, it flags dataset drift.
> Separately, if we have ground truth labels, it computes actual model metrics.
> My Airflow DAG runs this weekly. If F1 drops below 0.75, it automatically
> kicks off a retraining run and promotes the new model if it's better.
> This is what they call a closed-loop MLOps system."*

---

### Q: "How do you handle class imbalance? Only 26% of customers churn."

> *"I use SMOTE — Synthetic Minority Over-sampling Technique — exclusively on the
> training set, never on validation or test. SMOTE generates synthetic samples of
> the minority class by interpolating between existing minority examples.
> I also use `class_weight='balanced'` in LightGBM and `scale_pos_weight` in XGBoost
> as secondary mechanisms. I evaluate on F1 and Recall specifically because accuracy
> is misleading with imbalanced data — a model that predicts 'No churn' for everyone
> gets 73% accuracy but is completely useless."*

---

### Q: "What is data drift and why does it matter?"

> *"Data drift is when the statistical properties of your input features change over
> time, causing your model's predictions to degrade. Example: I train on customers
> where average monthly charges are $65. Six months later, the company raises prices
> and the average is $78. The model has never seen that distribution — its predictions
> become unreliable. The tricky part is it fails silently: the API keeps returning
> predictions, they're just wrong. You need active monitoring to catch it."*

---

### Q: "How is your API production-ready?"

> *"Several things: FastAPI with Pydantic validation means bad inputs get rejected
> before reaching the model with a clear 422 error. It logs every prediction with
> a UUID request ID. The `/health` endpoint means orchestrators like Kubernetes can
> do liveness checks. The model is loaded from the MLflow Registry at startup,
> so deploying a new model means promoting it in the Registry — no code change,
> no redeployment. Batch predict handles up to 500 customers in one call.
> Everything is containerised with Docker and the image is built in CI/CD."*

---

### Q: "Why did you use Great Expectations?"

> *"Most pipelines fail silently on bad data. Great Expectations gives you explicit
> contracts on what your data must look like — column presence, value ranges,
> cardinality, uniqueness, even statistical properties like churn rate bounds.
> If any expectation fails, the pipeline halts immediately with a clear error
> rather than propagating bad data to the model. During the interview I can
> show the data docs it generates — an HTML report of every check."*

---

### Q: "What's the difference between Staging and Production in MLflow?"

> *"It's a promotion workflow that mirrors software deployment. After training,
> the best model gets registered to Staging for validation — you can run integration
> tests, shadow traffic, or manual review. Once approved, it moves to Production.
> The old Production gets archived. My retraining DAG automates this:
> it only promotes to Production if the new model's F1 is higher than the current one.
> This prevents a bad retraining run from silently replacing a working model."*

---

## 📊 Things to Show During the Interview

Open these in separate browser tabs before the interview starts:

| Tab | URL | What to say |
|---|---|---|
| **MLflow UI** | http://localhost:5000 | "Here's every training run — hyperparams, metrics, model artifacts" |
| **API Docs** | http://localhost:8000/docs | "Live API — I can make a prediction right now" |
| **Streamlit** | http://localhost:8501 | "This is the monitoring dashboard the ops team would use" |
| **Airflow** | http://localhost:8080 | "These DAGs run automatically every Monday" |

---

## 📝 Resume Bullet Points (Copy These)

```
• Built production-grade MLOps pipeline for customer churn prediction
  (Airflow · DVC · MLflow · FastAPI · Evidently AI · Streamlit · Docker)

• Implemented automated hyperparameter tuning with Optuna TPE (50 trials/model),
  comparing Logistic Regression, XGBoost, and LightGBM with full MLflow tracking

• Engineered domain-driven features (tenure buckets, risk score, charge ratio)
  with SMOTE oversampling, improving F1 by 12% over baseline

• Deployed real-time prediction API with FastAPI serving MLflow Registry models;
  supports single and batch inference (up to 500 records)

• Built automated retraining system: Airflow DAG detects performance degradation
  via Evidently AI drift detection, triggers Optuna retraining, auto-promotes
  better model to Production — zero manual intervention required

• Achieved 94% test coverage across data, feature, model, and API layers
  with GitHub Actions CI/CD running on every push
```

---

## 🏆 What Makes This Stand Out

Most candidates have: *"I trained an XGBoost model and got 83% accuracy"*

You have:

| What Most Do | What You Did |
|---|---|
| Jupyter notebook | Production-grade modular codebase |
| GridSearch | Optuna (Bayesian, 10x faster) |
| No data validation | Great Expectations (automated contracts) |
| No versioning | DVC (reproducible, team-ready) |
| No tracking | MLflow (every experiment logged) |
| Flask script | FastAPI with Pydantic, batch, health check |
| No monitoring | Evidently AI + automated retraining |
| No CI/CD | GitHub Actions (lint, test, build, deploy) |
| Nothing containerised | Full Docker Compose stack |

**This is a system, not a script. That's the difference.**

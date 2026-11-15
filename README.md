# Real-Time Financial Fraud Detection System

An end-to-end machine learning and MLOps project for detecting potentially fraudulent credit card transactions through a REST API and interactive monitoring dashboard.

The system combines a class-weighted Random Forest classifier, transaction-level fraud risk scoring, SHAP explainability, fraud alert management, model versioning, and data drift monitoring.

## Key Features

- **Fraud Detection:** Random Forest classifier trained on highly imbalanced transaction data.
- **REST API:** FastAPI endpoint for real-time transaction predictions.
- **Interactive Dashboard:** Streamlit interface for transaction monitoring and model insights.
- **Explainable AI:** SHAP-based explanations showing which features influence individual predictions.
- **Fraud Investigation:** Alert creation, prioritization, assignment, resolution, and investigation notes.
- **Audit Trail:** Records changes to fraud alert statuses and investigation activity.
- **Data Drift Monitoring:** Population Stability Index (PSI) analysis to identify changes in feature distributions.
- **Model Registry:** Versioned model manifests, active model selection, and SHA-256 artifact integrity verification.
- **Structured Logging:** Rotating application logs for prediction activity, warnings, and exceptions.
- **Containerization:** Docker and Docker Compose for running the API and dashboard.
- **CI Pipeline:** GitHub Actions workflow for automated tests and Docker image builds.

## System Architecture

```mermaid
flowchart TD
    A[Transaction Input] --> B[FastAPI]
    B --> C[Feature Engineering]
    C --> D[Model Registry]
    D --> E[Random Forest Model]
    E --> F[Fraud Probability and Risk]
    F --> G[SQLite Database]
    G --> H[Streamlit Dashboard]
    H --> I[Fraud Investigation and Alerts]
    H --> J[SHAP Explanations]
    H --> K[Model and Drift Monitoring]
    B --> L[Structured Logging]
```

## Technology Stack

| Category | Technologies |
|---|---|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Imbalanced-learn |
| Explainable AI | SHAP |
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit, Plotly |
| Database | SQLite |
| Visualization | Matplotlib, Plotly |
| Model Persistence | Joblib |
| Testing | Pytest |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Version Control | Git |

## Dataset

The project uses the Credit Card Fraud Detection dataset containing anonymized European credit card transactions.

**Original dataset characteristics:**

- 284,807 transactions
- 31 original columns
- 492 fraudulent transactions
- 0.172% approximately fraudulent transactions
- Target variable: `Class` (1 = Fraud, 0 = Legitimate)

Dataset: [Credit Card Fraud Detection – Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Data Preparation

- Removed exact duplicate records, leaving 283,726 transactions.
- Preserved chronological ordering for model evaluation.
- Used chronological training, validation, and test splits.
- Applied training-only fitting for preprocessing transformations.
- Engineered additional temporal and transaction activity features.
- Addressed class imbalance using class-weighted Random Forest training.

The anonymized `V1`–`V28` variables are PCA-transformed features. They do not provide identifiable customer or cardholder information.

## Feature Engineering

The model uses 37 features, including:

- Original transaction time and anonymized PCA features.
- Cyclical time features: `hour_sin` and `hour_cos`.
- Elapsed-day information.
- Log-transformed and robust-scaled transaction amount features.
- Historical transaction activity counts over 60-second, 300-second, and 3600-second windows.

Historical activity features represent aggregate dataset activity, not individual cardholder transaction velocity.

## Model Development and Evaluation

A Random Forest classifier was selected for the final fraud detection model.

**Final model configuration:**

- Algorithm: RandomForestClassifier
- Number of estimators: 200
- Maximum depth: 12
- Minimum samples per leaf: 2
- Class weighting: `balanced_subsample`
- Random state: 42
- Model version: 1.0.0
- Decision threshold: 0.244112

The classification threshold was selected using validation data. The final test set was kept separate from threshold selection.

### Chronological Data Split

| Dataset | Transactions | Fraud Cases |
|---|---:|---:|
| Training | 198,609 | 366 |
| Validation | 42,560 | 55 |
| Testing | 42,557 | 52 |

### Held-Out Test Performance

| Metric | Result |
|---|---:|
| Precision | 79.59% |
| Recall | 75.00% |
| F1 Score | 77.23% |
| PR-AUC | 76.76% |
| ROC-AUC | 94.53% |
| Specificity | 99.98% |
| False Positive Rate | 0.024% |

### Confusion Matrix

| Actual / Predicted | Legitimate | Fraud |
|---|---:|---:|
| Legitimate | 42,495 | 10 |
| Fraud | 13 | 39 |

The model detected 39 of the 52 fraudulent transactions in the held-out test set.

**Evaluation note:** The test set contains only 52 fraudulent transactions. Metrics should therefore be interpreted with this limited fraud sample in mind. These results are experimental and do not establish production-level financial fraud detection performance.

## Explainable AI – SHAP

The project integrates SHAP to make model predictions more interpretable.

For individual transactions, the investigation interface provides:

- SHAP waterfall explanations.
- Top contributing features.
- Feature contribution visualizations.
- A detailed feature contribution table.

The SHAP explanation is designed to help investigators understand which model inputs contributed to a prediction. It does not establish the actual cause of a transaction.

## Fraud Alert Management

Fraudulent predictions can automatically generate investigation alerts.

Supported alert statuses:

- OPEN
- UNDER_REVIEW
- ESCALATED
- RESOLVED
- FALSE_POSITIVE

Additional functionality includes:

- Alert prioritization.
- Investigator assignment.
- Investigation notes.
- Resolution requirements.
- Alert history and audit events.
- Alert metrics and status tracking.

## Data Drift Monitoring

The system uses Population Stability Index (PSI) to compare reference and current feature distributions.

Drift categories:

| PSI | Interpretation |
|---|---|
| Below 0.10 | Stable |
| 0.10 to below 0.25 | Moderate drift |
| 0.25 or above | Significant drift |

The monitoring module supports:

- Feature-level drift analysis.
- Overall drift summaries.
- Offline reference-versus-test analysis.
- Live API transaction monitoring.
- Minimum sample-size validation.
- Missing-data and schema validation.

The offline comparison identified significant distribution differences between the training and later test periods. This indicates feature distribution shift, not necessarily a measured decline in predictive performance.

Live drift monitoring requires sufficient genuine API transactions and an available reference dataset.

## Model Registry and Versioning

The project includes a lightweight model registry.

Current registered model:

- Model ID: `fraud-random-forest`
- Version: `1.0.0`
- Algorithm: `RandomForestClassifier`
- Status: `ACTIVE`

The registry supports:

- Model manifest files.
- Active model pointer.
- Model configuration and evaluation metadata.
- Feature schema validation.
- SHA-256 model artifact verification.
- Model version information in API predictions.

## API Endpoints

The FastAPI application provides:

| Endpoint | Purpose |
|---|---|
| `GET /` | API health information |
| `GET /model-info` | Active model and evaluation information |
| `POST /predict` | Predict transaction fraud probability and risk |

Interactive API documentation is available at `/docs` when the API is running.

## Running the Project with Docker

### Prerequisites

- Docker Desktop
- Docker Compose

### 1. Clone the repository

```bash
git clone https://github.com/vinayydv2025-del/Real-Time-Financial-Fraud-Detection-System.git
cd "Real-Time Financial Fraud Detection System"
```

Replace the repository URL with the actual GitHub repository URL.

### 2. Build and start the services

```bash
docker compose up --build -d
```

### 3. Access the application

| Service | Local URL |
|---|---|
| Streamlit Dashboard | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Model Information | http://localhost:8000/model-info |

### 4. Check running containers

```bash
docker compose ps
```

### 5. View application logs

```bash
docker compose logs -f
```

### 6. Stop the services

```bash
docker compose down
```

The database is stored in a Docker named volume so transaction records persist across container restarts.

## Running Locally Without Docker

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API

```bash
uvicorn api.main:app --reload
```

### 4. Start the dashboard

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run app/streamlit_app.py
```

## Testing

The project includes automated tests for:

- API prediction and error handling.
- Structured logging.
- Database operations.
- Fraud alert management.
- Model registry validation.
- Model artifact integrity.
- Drift monitoring.
- Dashboard monitoring components.

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

**Latest local test result:** 54 passed.

## CI Pipeline

A GitHub Actions workflow is configured in:

```text
.github/workflows/ci.yml
```

The workflow is designed to run on pushes and pull requests.

It performs:

1. Repository checkout.
2. Python 3.12 setup.
3. Dependency installation.
4. Automated test execution.
5. Docker image build.

The workflow's actual GitHub Actions result should be verified after pushing the project to its GitHub repository.

## Project Structure

```text
Real-Time Financial Fraud Detection System/
│
├── api/
│   └── main.py
│
├── app/
│   ├── components/
│   │   ├── alerts.py
│   │   ├── investigation.py
│   │   ├── model_monitoring.py
│   │   ├── overview.py
│   │   └── simulator.py
│   ├── api_client.py
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── random_forest.joblib
│   └── registry/
│       ├── active_model.json
│       └── model_v1.0.0.json
│
├── reports/
│   ├── eda/
│   ├── explainability/
│   ├── feature_schema.json
│   └── final_test_evaluation.json
│
├── src/
│   ├── database.py
│   ├── drift_monitoring.py
│   ├── feature_engineering.py
│   ├── model_registry.py
│   ├── model_explainability.py
│   └── shap_service.py
│
├── tests/
│   ├── test_api.py
│   ├── test_api_client.py
│   ├── test_database.py
│   ├── test_drift_monitoring.py
│   ├── test_investigation.py
│   ├── test_model_monitoring.py
│   ├── test_model_registry.py
│   ├── test_overview.py
│   ├── test_simulator.py
│   └── test_simulator_ui.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Dashboard Screenshots

### 1. Executive Overview

The executive dashboard provides a high-level view of transaction activity, fraud alerts, detection rates, API latency, and active model information.

![Executive Overview 1](screenshots/Overview_01.png)

![Executive Overview 2](screenshots/Overview_02.png)

![Executive Overview 3](screenshots/Overview_03.png)

### 2. Fraud Investigation

The investigation dashboard provides transaction-level analysis, fraud probability, SHAP-based explanations, and feature contribution details.

![Fraud Investigation 1](screenshots/Fraud_Investigation_01.png)

![Fraud Investigation 2](screenshots/Fraud_Investigation_02.png)

![Fraud Investigation 3](screenshots/Fraud_Investigation_03.png)

![Fraud Investigation 4](screenshots/Fraud_Investigation_04.png)

![Fraud Investigation 5](screenshots/Fraud_Investigation_05.png)

![Fraud Investigation 6](screenshots/Fraud_Investigation_06.png)

![Fraud Investigation 7](screenshots/Fraud_Investigation_07.png)

### 3. Fraud Alert Management

The alert management interface supports investigation workflows, priority and status tracking, investigator assignment, resolution, and audit history.

![Fraud Alert Management 1](screenshots/Fraud_Alert_Management_01.png)

![Fraud Alert Management 2](screenshots/Fraud_Alert_Management_02.png)

![Fraud Alert Management 3](screenshots/Fraud_Alert_Management_03.png)

![Fraud Alert Management 4](screenshots/Fraud_Alert_Management_04.png)

### 4. Live Transaction Simulator

The simulator replays historical transactions through the FastAPI prediction service for sequential fraud detection.

![Live Transaction Simulator](screenshots/Live_Transaction_Simulator.png)

### 5. Model Performance and Monitoring

This section presents held-out test evaluation metrics, confusion matrix, model performance analysis, and feature drift monitoring.

![Model Performance 1](screenshots/Model_Performance_01.png)

![Model Performance 2](screenshots/Model_Performance_02.png)

![Model Performance 3](screenshots/Model_Performance_03.png)

![Model Performance 4](screenshots/Model_Performance_04.png)

![Model Performance 5](screenshots/Model_Performance_05.png)

![Model Performance 6](screenshots/Model_Performance_06.png)

## Limitations and Future Improvements

- The dataset contains anonymized features and does not provide real customer or cardholder identifiers.
- Historical activity features are aggregate activity measures, not cardholder-specific velocity features.
- The model is evaluated on a limited number of fraud cases in the held-out test set.
- Live drift monitoring requires a sufficient volume of genuine API transactions.
- In-memory API feature history resets when the API restarts and is not shared across multiple workers.
- The project has been verified locally using Docker; public cloud deployment and production-scale validation remain future improvements.

Potential future enhancements include:

- Cloud deployment and managed database integration.
- Persistent streaming feature computation.
- Model retraining and automated model promotion.
- Advanced monitoring and alert notifications.
- Authentication, authorization, and role-based investigator access.

## Author

Developed as an end-to-end machine learning and MLOps portfolio project, demonstrating practical implementation of fraud detection, explainable AI, REST API development, monitoring, model governance, testing, and containerization.
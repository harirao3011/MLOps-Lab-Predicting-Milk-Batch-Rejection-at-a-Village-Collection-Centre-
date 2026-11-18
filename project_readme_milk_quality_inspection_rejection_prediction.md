# Milk Quality Inspection & Rejection Prediction (End-to-End MLOps)

An end-to-end MLOps pipeline for automated dairy intake quality inspection and batch rejection risk prediction. Built with **Scikit-Learn**, **MLflow**, **FastAPI**, **Pydantic**, and **HTTPX**, this project demonstrates synthetic seasonal dataset versioning, domain feature engineering, experiment tracking, model registration, CI/CD input schema validation, and live REST API serving.

---

## 📋 Table of Contents
- [Problem Overview](#problem-overview)
- [System Architecture](#system-architecture)
- [Project Structure & Pipeline Stages](#project-structure--pipeline-stages)
- [Generated Datasets & Drift Analysis](#generated-datasets--drift-analysis)
- [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
- [Model Evaluation & MLflow Registry](#model-evaluation--mlflow-registry)
- [CI/CD Input Validation](#cicd-input-validation)
- [API Usage & Serving](#api-usage--serving)
- [Installation & Quickstart](#installation--quickstart)
- [Known Issues & Production Recommendations](#known-issues--production-recommendations)

---

## 🎯 Problem Overview
Raw milk batches arriving at dairy collection centers are highly perishable and sensitive to thermal exposure, acidity changes, and transit delays. Manual inspection struggles to evaluate interacting physicochemical and logistical risk factors at scale, leading to undetected spoilage or inconsistent rejections.

This system predicts whether an incoming batch should be rejected (`batch_rejected`: `Yes` / `No`) while addressing key production MLOps challenges:
- **Seasonal Covariate & Concept Drift:** Handling elevated ambient temperatures and extended transit delays during summer months.
- **Training-Serving Consistency:** Encapsulating feature engineering, scaling, and encoding inside a reusable transformer (`preprocessor_v1`) to prevent skew.
- **Model Governance:** Logging hyperparameters and evaluation metrics in **MLflow** and automatically registering the champion model (`MilkRejectionClassifier`).
- **Runtime Data Quality Gates:** Enforcing strict domain bounds on API payloads using **Pydantic** validators before model execution.

---

## 🏗️ System Architecture

```text
┌─────────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│  1. Data Versioning     │────▶│  2. Feature Pipeline    │────▶│  3. Model Training      │
│ • milk_data_v1 (Normal) │     │ • Thermal & Delay Index │     │ • Logistic Reg (F1:0.99)│
│ • milk_data_v2 (Summer) │     │ • Scaler + OHE (12-dim) │     │ • Random Forest (F1:0.96│
└─────────────────────────┘     └─────────────────────────┘     └────────────┬────────────┘
                                                                             │
┌─────────────────────────┐     ┌─────────────────────────┐     ┌────────────▼────────────┐
│  6. Live API & Probes   │◀────│  5. CI/CD Schema Guard  │◀────│  4. MLflow Registry     │
│ • POST /predict         │     │ • Pydantic Field Bounds │     │ • MilkRejectionClassifier│
│ • HTTPX Integration     │     │ • Unit Tests (Qty/Temp) │     │ • Champion Version 1    │
└─────────────────────────┘     └─────────────────────────┘     └─────────────────────────┘
```

---

## 📊 Generated Datasets & Drift Analysis
Two synthetic datasets (250 records each) are generated to simulate baseline operating conditions and seasonal summer drift. Ground-truth rejection labels (`Yes` / `No`) are determined when the composite spoilage score exceeds `1.5`:

$$\text{rejection\_score} = (6.7 - \text{measured\_acidity}) \times 3.0 + (\text{collection\_temp} - 4.0) \times 0.5 + (\text{transportation\_duration} \times 0.4) + (\text{cooling\_delay} \times 0.8) + ((\text{ambient\_temperature} - 20) \times 0.1)$$

| Dataset Version | File Name | Seed | Ambient Temp Mean | Transport Delay Mean | Rejected (`Yes`) | Accepted (`No`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Version 1 (Normal)** | `milk_data_v1.csv` | `42` | $18.0^\circ\text{C}$ ($\sigma=3$) | $1.5\text{ hrs}$ | 197 (78.8%) | 53 (21.2%) |
| **Version 2 (Summer)** | `milk_data_v2.csv` | `101` | $32.0^\circ\text{C}$ ($\sigma=3$) | $3.0\text{ hrs}$ | 248 (99.2%) | 2 (0.8%) |

### Input Features
- `milk_quantity` (Litres): Batch volume sampled from $\mathcal{U}(50, 1000)$
- `collection_temp` ($^\circ\text{C}$): Milk temperature at collection sampled from $\mathcal{U}(3.0, 8.0)$
- `transportation_duration` (Hours): Transit time sampled from $\text{Exp}(\text{scale})$
- `cooling_delay` (Hours): Delay before active refrigeration sampled from $\mathcal{U}(0.1, 3.0)$
- `milk_fat_percentage` (%): Fat content sampled from $\mathcal{U}(3.0, 5.5)$
- `measured_acidity` (pH): Acidity indicator sampled from $\mathcal{U}(6.4, 6.9)$
- `container_type` (Categorical): `Plastic Drum`, `Stainless Steel Tank`, or `Aluminum Can`
- `ambient_temperature` ($^\circ\text{C}$): External weather temperature sampled from $\mathcal{N}(\mu, 3)$

---

## ⚙️ Feature Engineering & Preprocessing
The `run_pipeline()` function transforms the 8 raw features into a standardized 12-dimensional feature matrix `(250, 12)`:
1. **Engineered Interaction Features:**
   - `temperature_exposure_index = collection_temp * ambient_temperature` (cumulative thermal stress)
   - `transport_delay_risk = transportation_duration * cooling_delay` (compounded microbial growth window)
2. **ColumnTransformer (`preprocessor_v1`):**
   - **Numerical Pipeline:** `StandardScaler()` applied to all 9 continuous features (7 raw + 2 engineered).
   - **Categorical Pipeline:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to `container_type` (3 encoded columns).

---

## 📈 Model Evaluation & MLflow Registry
Models are trained on an 80/20 split of `milk_data_v1` (`random_state=42`) and tracked under the MLflow experiment **`Milk_Batch_Inspection`**:

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Macro F1 | Registry Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **98.0%** | **0.98** | **1.00** | **0.99** | **0.96** | **Registered (`MilkRejectionClassifier` v1)** |
| **Random Forest** | 94.0% | 0.95 | 0.98 | 0.96 | 0.88 | Logged in MLflow |

- **Champion Selection:** `Logistic_Regression` achieved 100% recall on rejected batches (zero false negatives) and was automatically registered to the MLflow Model Registry as **`MilkRejectionClassifier` (Version 1)**.

---

## 🛡️ CI/CD Input Validation
Before serving, automated unit tests (`run_unit_tests()`) validate the Pydantic `MilkBatchInput` schema against invalid telemetry:
- **Negative Volume Guard:** Rejects `milk_quantity < 0` with a `ValueError`.
- **Physical Temperature Bounds Guard:** Rejects `collection_temp` or `ambient_temperature` outside `[-273.15, 100.0]` °C.
- **Container Type Enum Guard:** Restricts `container_type` strictly to valid storage vessel literals.

---

## 🚀 API Usage & Serving
The FastAPI service (`Milk Quality Inspection Service`) runs via `uvicorn` on `http://127.0.0.1:8000` and exposes `POST /predict`.

### Example Request (`POST /predict`)
```json
{
  "milk_quantity": 250.0,
  "collection_temp": 3.5,
  "transportation_duration": 0.5,
  "cooling_delay": 0.2,
  "milk_fat_percentage": 4.1,
  "measured_acidity": 6.8,
  "container_type": "Stainless Steel Tank",
  "ambient_temperature": 15.0
}
```

### Example Response
```json
{
  "prediction": "Likely Accepted"
}
```
*(High-risk batches return `{"prediction": "Rejection Risk"}`.)*

---

## 💻 Installation & Quickstart

1. **Install Dependencies:**
   ```bash
   pip install numpy pandas scikit-learn mlflow fastapi uvicorn pydantic nest-asyncio httpx
   ```

2. **Run the Notebook / Pipeline:**
   Open `Untitled35.ipynb` in Jupyter or Google Colab and execute all cells sequentially to:
   - Generate `milk_data_v1.csv` and `milk_data_v2.csv`
   - Fit the preprocessing pipeline and train candidate models
   - Log runs and register `MilkRejectionClassifier` in MLflow
   - Execute CI/CD Pydantic validation tests
   - Launch the local FastAPI server and run `httpx` endpoint checks

3. **View MLflow UI (Optional):**
   ```bash
   mlflow ui --port 5000
   ```

---

## 🔧 Known Issues & Production Recommendations
1. **Inference Target Column `KeyError` Fix:** In the notebook's `run_pipeline()` helper, `y = df_feat['batch_rejected'].map(...)` executes unconditionally even when `is_train=False`, causing `POST /predict` requests (which omit `batch_rejected`) to return HTTP 500 (`{"detail": "'batch_rejected'"}`). Guarding target extraction with `if 'batch_rejected' in df_feat.columns:` resolves this issue for live inference.
2. **Unified Scikit-Learn Pipeline Artifact:** Bundle `preprocessor_v1` and the winning `LogisticRegression` estimator inside a single `sklearn.pipeline.Pipeline` prior to `mlflow.sklearn.log_model()` so a single registered artifact handles raw feature ingestion end-to-end.
3. **Automated Drift Retraining:** Monitor incoming production feature distributions against `milk_data_v1.csv` baselines to trigger automated retraining when summer temperature drift occurs.
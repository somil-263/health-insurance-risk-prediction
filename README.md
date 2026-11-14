# 🏥 Health Insurance Underwriting Risk Predictor

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20App-blue)](https://huggingface.co/spaces/somil263/health-insurance-risk-predictor)
![Python](https://img.shields.io/badge/Python-3.10+-yellow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-brightgreen)

An end-to-end Machine Learning regression system designed for health insurance underwriting. The model predicts a patient's continuous **Underwriting Risk Score (0.0 to 1.0)** using clinical vitals and lifestyle indicators, deployed as an interactive web application on Hugging Face Spaces.

🔗 **Live Interactive App:** [Health Insurance Risk Predictor](https://huggingface.co/spaces/somil263/health-insurance-risk-predictor)

---

## 📌 Problem Statement & Business Context
In health insurance underwriting, accurately assessing applicant risk before policy issuance is critical for pricing and risk management. Using a dataset of **100,000 patient records**, this project builds a leakage-free predictive pipeline that evaluates core medical vitals and lifestyle habits to generate real-time risk assessments.

---

## 🛠️ Data Engineering & Leakage Prevention
- **Dataset Scale:** 100,000 rows × 54 raw demographic, lifestyle, clinical, and policy features.
- **Missing Value Treatment:** Identified and imputed **30,083 missing values** in `alcohol_freq` as a distinct `'None'` category rather than dropping records.
- **Strict Target & Post-Underwriting Leakage Removal:** Removed post-claim and derived target variables (`is_high_risk`, `annual_medical_cost`, `annual_premium`, `monthly_premium`, `total_claims_paid`, `avg_claim_amount`, `claims_count`, and `person_id`) prior to training to ensure 100% genuine predictive generalization.
- **Feature Selection for Deployment:** Streamlined the production model to **8 high-impact, user-accessible inputs**: `age`, `bmi`, `systolic_bp`, `ldl`, `hba1c`, `chronic_count`, `smoker`, and `alcohol_freq`.

---

## 📊 Model Performance
Trained and evaluated multiple regression architectures (`LinearRegression` baseline vs. tree-based ensembles) using an 80/20 train-test split:

| Model | Features Used | Test $R^2$ Score | Status |
| :--- | :--- | :---: | :--- |
| **HistGradientBoostingRegressor** | Full Clean Feature Set | `0.9999` | Experimented |
| **HistGradientBoostingRegressor (Production)** | **8 Core Clinical & Lifestyle Features** | **`0.9998`** | **Deployed** |

---

## 📂 Repository Structure
```text
├── notebook.ipynb        # End-to-end EDA, preprocessing, feature engineering & model evaluation
├── app.py                # Gradio web application interface
├── risk_model.joblib     # Serialized HistGradientBoostingRegressor model & feature schema
├── requirements.txt      # Production dependencies
└── README.md             # Project documentation

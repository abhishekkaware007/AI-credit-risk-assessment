# 💎 CreditIQ — AI Credit Risk Assessment

Predicts loan default probability using machine learning.
Classifies borrowers into 5 risk tiers with full explainability.

---

## 🚦 Risk Categories

| Level | Probability | Decision |
|---|---|---|
| 🟢 MINIMAL | 0–20% | Approve |
| 🟡 LOW | 20–40% | Approve with review |
| 🟠 MODERATE | 40–60% | Enhanced review |
| 🔴 HIGH | 60–80% | Reject |
| 🚨 CRITICAL | 80–100% | Immediate rejection |

---

## ✨ Features

- **Real-time prediction** — Enter applicant details, get instant risk score
- **SHAP explainability** — Shows exactly why the model made each decision
- **What-If Analysis** — Shows what changes would reduce the applicant's risk
- **Batch prediction** — Upload CSV of multiple applicants, get all scores at once
- **Assessment history** — Save, view and download all past assessments

---

## 📊 Model Performance

| Model | AUC-ROC | Accuracy |
|---|---|---|
| **Random Forest** | **0.9231** | **94.59%** |
| Gradient Boosting | 0.8421 | 78.89% |
| XGBoost | 0.8280 | 82.42% |
| Logistic Regression | 0.7988 | 78.41% |

---

## 🛠 Tech Stack

Python · Scikit-learn · Random Forest · SHAP
Streamlit · Plotly · Pandas · NumPy · XGBoost

---

## 🚀 Run Locally

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/CreditRiskProject.git
cd CreditRiskProject/GiveMeSomeCredit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Structure


CreditRiskProject/
│
├── GiveMeSomeCredit/
│   ├── app.py
│   ├── requirements.txt
│   └── notebooks/
│       ├── 01_EDA.ipynb
│       ├── 02_preprocessing.ipynb
│       ├── 03_model_training.ipynb
│       └── 04_evaluation.ipynb
│
└── models/
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── gradient_boosting.pkl
    └── logistic_regression.pkl

---

## 📋 Dataset

Kaggle — Give Me Some Credit
150,000 real borrower records
Target: Did the person default within 2 years?

---

## 👤 Author

**Abhishek Kaware** · VIT Pune · 2025

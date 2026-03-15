# 💳 End-to-End Credit Risk Modeling & Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://credit-risk-project-dbpfuopxmaezrtgbveundv.streamlit.app/)
> **Live Demo:** [Access the Interactive Credit Risk Dashboard](https://credit-risk-project-dbpfuopxmaezrtgbveundv.streamlit.app/)

### 📝 Project Overview
This project demonstrates a professional-grade Credit Risk / Credit Card Approval system—from synthetic data generation to cloud deployment. In the banking sector, the cost of a **False Negative** (approving a defaulter) is significantly higher than a **False Positive**. This system is engineered to prioritize **Recall** and **Business Logic** over raw accuracy.

---

**📂 Table of Contents**
1. [Problem Statement](#-problem-statement)
2. [Tech Stack](#-tech-stack)
3. [Data Dictionary](#-data-dictionary)
4. [Advanced Machine Learning Workflow](#-advanced-machine-learning-workflow)
5. [Model Evaluation & Business Logic](#-model-evaluation--business-logic)
6. [Explainability & Regulatory Compliance](#-explainability--regulatory-compliance)
7. [How to Run](#-how-to-run)

---

### 🎯 Problem Statement
The primary challenge in credit modeling is balancing **Profitability** (approving safe customers) vs. **Risk** (avoiding defaulters). Because high-risk customers are a minority, this project implements:
1. **Recall-Optimized Modeling:** Identifying as many potential defaulters as possible.
2. **Cost-Sensitive Learning:** Accounting for the $5\times$ higher cost of credit defaults compared to missed opportunities.

### 🛠️ Tech Stack
* **Language:** Python 3.9+
* **ML Frameworks:** XGBoost, Scikit-learn
* **Imbalance Handling:** Imbalanced-learn (SMOTE)
* **Deployment:** Streamlit Cloud
* **Analysis:** Pandas, NumPy, Matplotlib, Seaborn

### 📊 Data Dictionary
We simulate realistic banking-style data including:
* **Age/Income/Employment:** Basic demographics and stability indicators.
* **Credit Score:** Bureau-provided score (300-850).
* **Utilization Ratio:** Percentage of credit currently used.
* **Debt-to-Income (DTI):** Calculated as $\frac{\text{Requested Loan}}{\text{Annual Income}}$.
* **Late Payments:** Number of times a customer missed a due date.

---

### ⚙️ Advanced Machine Learning Workflow

#### 1. Handling Class Imbalance (SMOTE)
Standard models often ignore minority "High Risk" classes. We utilize **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic samples, ensuring the model learns the specific signatures of default behavior rather than just guessing the majority class.

#### 2. Feature Engineering
We translated banking domain knowledge into engineered features:
* `high_utilization`: Flags users using $>75\%$ of their limit.
* `low_credit_score`: Binary indicator for scores below 600.
* `stable_job`: Indicator for $>5$ years of employment.

#### 3. Threshold Optimization (The 0.35 Rule)
Instead of a default 0.5 threshold, we use **0.35**. This is because the "Business Cost" of a False Negative (approving a defaulter) is significantly higher than a False Positive.

---

### 📈 Model Evaluation & Business Logic
We compared multiple models based on their **Business Cost** and **ROC-AUC**:

| Model | ROC-AUC | Recall | Business Cost | Explainability |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | Medium | Medium | Medium | High |
| **XGBoost (Selected)** | **Very High** | **Very High** | **Lowest** | **Medium** |

**Why XGBoost?** It provided the best rank-ordering of risk and the highest recall for catching risky customers.

---

### 🔍 Explainability & Regulatory Compliance
To meet banking regulatory requirements (like GDPR or FCRA), we provide **Decision Reasoning**. When a customer is rejected, the app displays specific reasons such as "High DTI" or "Low Credit Score" to ensure transparency.

---

### 🚦 How to Run

1. **Clone & Install:** ```bash
   git clone [https://github.com/iambhavishya/your-repo-name.git](https://github.com/iambhavishya/your-repo-name.git)
   cd your-repo-name
   pip install -r requirements.txt

# 💳 End-to-End Credit Risk Modeling & Deployment

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://credit-risk-project-dbpfuopxmaezrtgbveundv.streamlit.app/)
> **Live Demo:** [Access the Interactive Credit Risk Dashboard](https://credit-risk-project-dbpfuopxmaezrtgbveundv.streamlit.app/)

### 📝 Project Overview
This project demonstrates a production-ready Credit Risk / Credit Card Approval system—from synthetic data generation to cloud deployment. In banking, the financial cost of a **False Negative** (approving a defaulter) is significantly higher than a **False Positive**. This system is specifically engineered to optimize **Recall** and **Business Decision Logic**.

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
The primary challenge in credit modeling is balancing **Profitability** (approving safe customers) vs. **Risk** (avoiding defaulters). Because high-risk customers are a minority in real-world datasets, this project implements:
* **Recall-Optimized Modeling:** Prioritizing the detection of potential defaulters.
* **Cost-Sensitive Learning:** Accounting for the $5\times$ higher cost of credit defaults compared to missed opportunities.

### 🛠️ Tech Stack
* **Language:** Python 3.9+
* **ML Frameworks:** XGBoost, Scikit-learn
* **Imbalance Handling:** Imbalanced-learn (SMOTE)
* **Deployment:** Streamlit Cloud
* **Analysis:** Pandas, NumPy, Matplotlib, Seaborn

### 📊 Data Dictionary
The model utilizes realistic banking-style features:
* **Age/Income/Employment:** Basic demographics and financial stability.
* **Credit Score:** Bureau-provided scores (300-850).
* **Utilization Ratio:** Percentage of total credit currently utilized.
* **Debt-to-Income (DTI):** Calculated as $\frac{\text{Requested Loan}}{\text{Annual Income}}$.
* **Late Payments:** Historical count of missed due dates.

---

### ⚙️ Advanced Machine Learning Workflow

#### 1. Handling Class Imbalance (SMOTE)
To prevent the model from being biased toward the majority "Low Risk" class, we utilize **SMOTE** (Synthetic Minority Over-sampling Technique) during training to learn the distinct patterns of high-risk behavior.

#### 2. Feature Engineering
We engineered specific business-driven features to improve the signal:
* `high_utilization`: Flags users using $>75\%$ of their credit limit.
* `low_credit_score`: Binary indicator for scores below 600.
* `stable_job`: Indicator for $>5$ years of employment.

#### 3. Threshold Optimization (The 0.35 Rule)
Instead of a default 0.5 probability threshold, we use **0.35**. This adjustment ensures a higher sensitivity (Recall) toward risky customers, aligning with conservative banking risk appetite.

---

### 📈 Model Evaluation & Business Logic
We compared multiple models based on their **Business Cost** and **ROC-AUC**:

| Model | ROC-AUC | Recall | Business Cost | Explainability |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.82 | 0.74 | Medium | High |
| **XGBoost (Selected)** | **0.91** | **0.88** | **Lowest** | **Medium** |

**Why XGBoost?** It provided the best rank-ordering of risk and captured non-linear relationships that linear models missed.

---

### 🔍 Explainability & Regulatory Compliance
To meet banking regulatory requirements (like GDPR or FCRA), the deployment provides **Automated Decision Reasoning**. When a customer is rejected, the app displays transparent reasons such as "High DTI" or "Low Credit Score" to ensure the decision-making process is not a "black box."

---

### 🚦 How to Run

1. **Clone & Install:**
   ```bash
   git clone [https://github.com/iambhavishya/your-repo-name.git](https://github.com/iambhavishya/your-repo-name.git)
   cd your-repo-name
   pip install -r requirements.txt

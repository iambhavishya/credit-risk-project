# **💳 End-to-End Credit Risk Modeling & Deployment**

### **📝 Project Overview**

This project demonstrates a full-fledged Credit Risk / Credit Card Approval system exactly how a professional data scientist would build it—from synthetic data generation to deployment. Financial institutions must decide whether to approve a credit card for a customer while minimizing the risk of default. The goal is to predict whether a customer is **Low Risk (0)** or **High Risk (1)** and use this prediction for real-time decision-making.

### ---

**📂 Table of Contents**

1. [Problem Statement](https://www.google.com/search?q=%23problem-statement&authuser=1)  
2. [Tech Stack](https://www.google.com/search?q=%23tech-stack&authuser=1)  
3. [Data Dictionary](https://www.google.com/search?q=%23data-dictionary&authuser=1)  
4. [Advanced Machine Learning Workflow](https://www.google.com/search?q=%23advanced-machine-learning-workflow&authuser=1)  
5. [Model Evaluation & Business Logic](https://www.google.com/search?q=%23model-evaluation--business-logic&authuser=1)  
6. [How to Run](https://www.google.com/search?q=%23how-to-run&authuser=1)  
7. [Explainability & Regulatory Compliance](https://www.google.com/search?q=%23explainability--regulatory-compliance&authuser=1)

### ---

**🎯 Problem Statement**

The primary challenge in credit modeling is balancing **Profitability** (approving safe customers) vs. **Risk** (avoiding defaulters). High-risk customers are a minority in the real world, making accuracy a misleading metric. This project focuses on **Recall** to catch as many risky customers as possible.

### **🛠️ Tech Stack**

* **Language:** Python

* **Data Handling:** pandas, numpy

* **Visualization:** matplotlib, seaborn

* **Machine Learning:** scikit-learn, xgboost

* **Imbalance Handling:** imbalanced-learn (SMOTE)

* **Deployment:** Streamlit

### **📊 Data Dictionary**

We simulate realistic banking-style data including:

* **Age/Income/Employment:** Basic demographics and stability indicators.

* **Credit Score:** Bureau-provided score (300-850).

* **Utilization Ratio:** Percentage of credit currently used.

* **Debt-to-Income (DTI):** Calculated as (Requested Loan / Annual Income).

* **Late Payments:** Number of times a customer missed a due date.

### ---

**⚙️ Advanced Machine Learning Workflow**

#### **1\. Handling Class Imbalance (SMOTE)**

In credit risk, defaulters are a minority. We use **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the training set, allowing the model to learn the distinct patterns of high-risk behavior.

#### **2\. Feature Engineering**

We engineered specific business-driven features to improve signal:

* high\_utilization: Flags users using \>75% of their limit.

* low\_credit\_score: Binary indicator for scores below 600\.

* stable\_job: Indicator for \>5 years of employment.

#### **3\. Threshold Optimization (The 0.35 Rule)**

Instead of a default 0.5 threshold, we use **0.35**. This is because the "Business Cost" of a False Negative (approving a defaulter) is roughly **5x higher** than a False Positive.

### ---

**📈 Model Evaluation & Business Logic**

We compared multiple models based on their **Business Cost** and **ROC-AUC**:

| Model | ROC-AUC | Recall | Business Cost | Explainability |
| :---- | :---- | :---- | :---- | :---- |
| Logistic Regression | Medium | Medium | Medium | High  |
| **XGBoost (Selected)** | **Very High** | **Very High** | **Lowest** |  **Medium**  |

**Why XGBoost?** It provided the best rank-ordering of risk and the highest recall for catching risky customers.

### ---

**🔍 Explainability & Regulatory Compliance**

To meet banking regulatory requirements (like GDPR or FCRA), we provide **Decision Reasoning**. When a customer is rejected, the app displays specific reasons such as "High DTI" or "Low Credit Score".

### ---

**🚦 How to Run**

1. **Clone & Install:**  
   Bash  
   pip install \-r requirements.txt

2. **Generate Data:**  
   Bash  
   python src/data\_generation.py

3. **Train Pipeline:**  
   Bash  
   python src/train\_pipeline.py

4. **Run Streamlit App:**  
   Bash  
   streamlit run app/streamlit\_app.py


from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. PAGE CONFIG (MUST BE THE VERY FIRST ST COMMAND)
st.set_page_config(page_title="Credit Risk Pro", layout="centered")

# 2. REQUIRED FUNCTION FOR PIPELINE LOADING
# This must match the training logic exactly [cite: 63, 65, 66, 67]
def add_engineered_features(df):
    X = df.copy()
    X['high_utilization'] = (X['utilization_ratio'] > 0.75).astype(int)
    X['low_credit_score'] = (X['credit_score'] < 600).astype(int)
    X['stable_job'] = (X['employment_years'] > 5).astype(int)
    return X

# 3. ROBUST PATH LOADING
BASE_DIR = Path(__file__).resolve().parents[1]
pipeline_path = BASE_DIR / "data" / "processed" / "credit_risk_pipeline.pkl"

# Load the industry-grade XGBoost pipeline [cite: 125, 128]
pipeline = joblib.load(pipeline_path)

# 4. APP HEADER
st.title('💳 Credit Risk Assessment Pro (XGBoost)')
st.write("Production-ready pipeline with SMOTE balancing and Decision Logic. [cite: 177]")

# 5. INPUTS (Matches guide Page 1-2) [cite: 15, 17]
col1, col2 = st.columns(2)
with col1:
    age = st.slider('Age', 21, 65, 30)
    income = st.number_input('Annual Income (₹)', 20000, 200000, 50000)
    employment_years = st.slider('Employment Years', 0, 40, 5)
    credit_score = st.slider('Credit Score', 300, 850, 700)
with col2:
    existing_loans = st.slider('Existing Loans', 0, 5, 1)
    loan_amount = st.number_input('Requested Credit (₹)', 1000, 50000, 15000)
    utilization_ratio = st.slider('Utilization Ratio', 0.0, 1.0, 0.3)
    late_payments = st.slider('Late Payments', 0, 10, 0)

# Derived feature [cite: 37, 154]
debt_to_income = loan_amount / income

# 6. PREDICTION LOGIC
if st.button('Check Credit Risk'):
    input_data = pd.DataFrame([[age, income, employment_years, credit_score, 
                                existing_loans, loan_amount, utilization_ratio, 
                                late_payments, debt_to_income]], 
                              columns=['age', 'income', 'employment_years', 'credit_score', 
                                       'existing_loans', 'loan_amount', 'utilization_ratio', 
                                       'late_payments', 'debt_to_income'])
    
    # Predict probability
    risk_prob = pipeline.predict_proba(input_data)[0][1]
    
    # Business Decision Threshold (Optimized) [cite: 132, 212]
    threshold = 0.35 
    
    st.divider()
    
    if risk_prob < threshold:
        st.success(f"✅ Approved | Risk Probability: {risk_prob:.2%}")
    else:
        st.error(f"❌ Rejected | Risk Probability: {risk_prob:.2%}")
        
        # 7. DECISION EXPLANATION (Regulatory Requirement) [cite: 197, 236]
        st.subheader('Approval Reasoning')
        reasons = []
        if credit_score < 600: reasons.append("Low credit bureau score [cite: 40]")
        if late_payments > 3: reasons.append("Excessive late payments [cite: 41]")
        if utilization_ratio > 0.8: reasons.append("High credit utilization [cite: 42]")
        if debt_to_income > 0.6: reasons.append("High debt-to-income ratio [cite: 43]")
        
        if reasons:
            for r in reasons:
                st.write(f"- {r}")

st.divider()
st.caption("Industry-Grade Stack: XGBoost + SMOTE + Threshold Optimization")
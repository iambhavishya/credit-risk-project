from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. PAGE CONFIG
st.set_page_config(page_title="Credit Risk Pro", layout="centered", page_icon="💳")

# 2. REQUIRED FUNCTION FOR PIPELINE LOADING
# This function handles the automated feature engineering during inference
def add_engineered_features(df):
    X = df.copy()
    X['high_utilization'] = (X['utilization_ratio'] > 0.75).astype(int)
    X['low_credit_score'] = (X['credit_score'] < 600).astype(int)
    X['stable_job'] = (X['employment_years'] > 5).astype(int)
    return X

# 3. ROBUST PATH LOADING
# Adjusting path to locate the model in your project structure
BASE_DIR = Path(__file__).resolve().parents[1]
pipeline_path = BASE_DIR / "data" / "processed" / "credit_risk_pipeline.pkl"

@st.cache_resource
def load_model():
    return joblib.load(pipeline_path)

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 4. APP HEADER
st.title('💳 Credit Risk Assessment Pro')
st.markdown("""
### End-to-End Decision Support System
This application uses an **XGBoost** classifier trained on synthetic banking data. 
It features **SMOTE** for class balancing and a custom **0.35 probability threshold** to prioritize risk detection.
""")

st.divider()

# 5. INPUTS
st.subheader("Customer Financial Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 21, 65, 30)
    income = st.number_input('Annual Income (₹)', 20000, 200000, 50000, step=5000)
    employment_years = st.slider('Employment Years', 0, 40, 5)
    credit_score = st.slider('Credit Score', 300, 850, 700)

with col2:
    existing_loans = st.slider('Existing Loans', 0, 10, 1)
    loan_amount = st.number_input('Requested Credit (₹)', 1000, 50000, 15000, step=1000)
    utilization_ratio = st.slider('Utilization Ratio (Credit Used)', 0.0, 1.0, 0.3)
    late_payments = st.slider('Late Payments (Last 2 Years)', 0, 20, 0)

# Derived feature
debt_to_income = loan_amount / income

# 6. PREDICTION LOGIC
if st.button('Evaluate Credit Risk', use_container_width=True):
    # Prepare data for the pipeline
    input_data = pd.DataFrame([[
        age, income, employment_years, credit_score, 
        existing_loans, loan_amount, utilization_ratio, 
        late_payments, debt_to_income
    ]], columns=[
        'age', 'income', 'employment_years', 'credit_score', 
        'existing_loans', 'loan_amount', 'utilization_ratio', 
        'late_payments', 'debt_to_income'
    ])
    
    # Predict probability using the XGBoost Pipeline
    risk_prob = pipeline.predict_proba(input_data)[0][1]
    
    # Business Decision Threshold (Optimized for Recall)
    threshold = 0.35 
    
    st.divider()
    
    # Display Result
    if risk_prob < threshold:
        st.success(f"### ✅ APPROVED\n**Risk Probability:** {risk_prob:.2%}")
        st.balloons()
    else:
        st.error(f"### ❌ REJECTED\n**Risk Probability:** {risk_prob:.2%}")
        
        # 7. DECISION EXPLANATION (Regulatory Requirement)
        st.subheader('Automated Decision Reasoning')
        reasons = []
        if credit_score < 600: reasons.append("Credit score falls below the minimum internal threshold.")
        if late_payments > 3: reasons.append("History of excessive late payments detected.")
        if utilization_ratio > 0.8: reasons.append("High credit utilization indicates potential liquidity stress.")
        if debt_to_income > 0.6: reasons.append("Debt-to-Income ratio exceeds safe lending limits.")
        
        if reasons:
            for r in reasons:
                st.info(f"👉 {r}")
        else:
            st.info("👉 The rejection is based on complex feature interactions within the XGBoost model.")

st.divider()
st.caption("Developed by iambhavishya | Tech Stack: Python, XGBoost, Scikit-learn, Streamlit")

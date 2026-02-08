import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# 1. Feature Engineering Logic (Must match data_generation.py)
def add_engineered_features(df):
    X = df.copy()
    X["low_credit_score"] = (X["credit_score"] < 650).astype(int)
    X["high_dti"] = (X["debt_to_income"] > 0.35).astype(int)
    X["high_utilization"] = (X["credit_utilization"] > 0.7).astype(int)
    X["unstable_employment"] = (X["employment_years"] < 2).astype(int)
    return X

# 2. Load Data
# Assuming you are running from the /src folder
X_train, X_test, y_train, y_test = joblib.load("../data/processed/dataset.pkl")

# 3. Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
}

results = {}

for name, model in models.items():
    # Create a temporary pipeline for each model
    temp_pipe = Pipeline([
        ("engineering", FunctionTransformer(add_engineered_features)),
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])
    
    temp_pipe.fit(X_train, y_train)
    y_prob = temp_pipe.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    results[name] = roc_auc
    print(f"{name} ROC-AUC: {roc_auc:.4f}")

# Save the best raw model
best_model_name = max(results, key=results.get)
joblib.dump(models[best_model_name], "../data/processed/best_model.pkl")
print(f"✅ Best Model Saved: {best_model_name}")
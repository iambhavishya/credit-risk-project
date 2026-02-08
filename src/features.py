import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib

# Load raw data
df = pd.read_csv("data/raw/credit_data.csv")

# Feature engineering
df["low_credit_score"] = (df["credit_score"] < 600).astype(int)
df["high_dti"] = (df["debt_to_income"] > 0.4).astype(int)
df["high_utilization"] = (df["credit_utilization"] > 0.8).astype(int)
df["unstable_employment"] = (df["employment_years"] < 2).astype(int)

# Separate features and target
X = df.drop("default", axis=1)
y = df["default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale numeric features
numeric_cols = [
    "age", "income", "employment_years", "credit_score",
    "existing_loans", "missed_payments",
    "debt_to_income", "credit_utilization"
]

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Save processed data & scaler
Path("data/processed").mkdir(parents=True, exist_ok=True)

joblib.dump((X_train, X_test, y_train, y_test), "data/processed/dataset.pkl")
joblib.dump(scaler, "data/processed/scaler.pkl")

print("✅ Features engineered and data prepared")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

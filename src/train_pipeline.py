import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Must match app.py exactly [cite: 63, 64, 65]
def add_engineered_features(df):
    X = df.copy()
    X['high_utilization'] = (X['utilization_ratio'] > 0.75).astype(int)
    X['low_credit_score'] = (X['credit_score'] < 600).astype(int)
    X['stable_job'] = (X['employment_years'] > 5).astype(int)
    return X

df = pd.read_csv("data/raw/credit_data.csv")
X = df.drop('risk', axis=1)
y = df['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# SMOTE for Class Imbalance [cite: 176, 179]
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

pipeline = Pipeline([
    ('engineering', FunctionTransformer(add_engineered_features)),
    ('scaler', StandardScaler()),
    ('model', XGBClassifier(eval_metric='logloss', random_state=42))
])

pipeline.fit(X_train_res, y_train_res)
joblib.dump(pipeline, "data/processed/credit_risk_pipeline.pkl")
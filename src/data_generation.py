import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
n = 5000

data = pd.DataFrame({
    'age': np.random.randint(21, 65, n),
    'income': np.random.normal(60000, 20000, n).clip(20000, 200000),
    'employment_years': np.random.randint(0, 40, n),
    'credit_score': np.random.randint(300, 850, n),
    'existing_loans': np.random.randint(0, 6, n),
    'loan_amount': np.random.normal(15000, 8000, n).clip(1000, 50000),
    'utilization_ratio': np.random.uniform(0.1, 1.0, n),
    'late_payments': np.random.poisson(1.5, n),
})

data['debt_to_income'] = (data['loan_amount'] / data['income']).clip(0, 2)

# Professional Risk Logic [cite: 38, 42]
data['risk'] = (
    (data['credit_score'] < 600) |
    (data['late_payments'] > 3) |
    (data['utilization_ratio'] > 0.8) |
    (data['debt_to_income'] > 0.6)
).astype(int)

output_path = Path("data/raw/credit_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(output_path, index=False)
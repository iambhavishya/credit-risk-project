import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("data/raw/credit_data.csv")

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["default"].value_counts(normalize=True))

# Basic statistics
print("\nSummary statistics:")
print(df.describe())

# Plot target distribution
plt.figure()
sns.countplot(x="default", data=df)
plt.title("Default vs Non-Default Distribution")
plt.show()

# Credit score vs default
plt.figure()
sns.boxplot(x="default", y="credit_score", data=df)
plt.title("Credit Score by Default Status")
plt.show()

# Debt-to-income vs default
plt.figure()
sns.boxplot(x="default", y="debt_to_income", data=df)
plt.title("Debt-to-Income Ratio by Default Status")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

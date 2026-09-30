"""
Employee Data Analysis and Supervised ML Model Comparison
Internship: AI Pioneers (Machine Learning) - Yuva Intern

Steps:
1. Create a sample employee dataset (with some missing values and duplicates)
2. Clean and preprocess the data using Pandas and NumPy
3. Train two supervised models with Scikit-learn (Linear Regression, Decision Tree)
4. Compare the models using MSE and R2 score
5. Save graphs using Matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Create sample dataset
# ---------------------------------------------------------------
n = 300
departments = ["IT", "HR", "Finance", "Sales", "Marketing"]
cities = ["Chennai", "Bangalore", "Hyderabad", "Mumbai"]

df = pd.DataFrame({
    "employee_id": range(1001, 1001 + n),
    "age": np.random.randint(22, 58, n),
    "experience_years": np.random.randint(0, 30, n),
    "department": np.random.choice(departments, n),
    "city": np.random.choice(cities, n),
    "education_level": np.random.choice([1, 2, 3], n, p=[0.4, 0.4, 0.2]),
})
df["experience_years"] = np.minimum(df["experience_years"], df["age"] - 21)

dept_bonus = {"IT": 15000, "HR": 5000, "Finance": 12000, "Sales": 8000, "Marketing": 7000}
df["salary"] = (
    20000
    + df["experience_years"] * 2500
    + df["education_level"] * 6000
    + df["department"].map(dept_bonus)
    + np.random.normal(0, 4000, n)
).round(0)

# add some messy data on purpose (missing values and duplicates)
df.loc[np.random.choice(n, 15, replace=False), "experience_years"] = np.nan
df.loc[np.random.choice(n, 10, replace=False), "age"] = np.nan
df.loc[np.random.choice(n, 8, replace=False), "city"] = np.nan
df = pd.concat([df, df.sample(6, random_state=1)], ignore_index=True)

df.to_csv("employee_data.csv", index=False)
print("Raw data shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------
# 2. Data preprocessing
# ---------------------------------------------------------------
df = df.drop_duplicates().reset_index(drop=True)
df["age"] = df["age"].fillna(df["age"].median())
df["experience_years"] = df["experience_years"].fillna(df["experience_years"].mean())
df["city"] = df["city"].fillna(df["city"].mode()[0])

df_clean = df.drop(columns=["employee_id"])
df_clean = pd.get_dummies(df_clean, columns=["department", "city"], drop_first=True)
df_clean.to_csv("employee_data_cleaned.csv", index=False)
print("\nClean data shape:", df_clean.shape)

X = df_clean.drop(columns=["salary"])
y = df_clean["salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 3. Train two supervised models
# ---------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)
dt_pred = dt.predict(X_test_scaled)

# ---------------------------------------------------------------
# 4. Compare the models
# ---------------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree"],
    "MSE": [mean_squared_error(y_test, lr_pred), mean_squared_error(y_test, dt_pred)],
    "RMSE": [np.sqrt(mean_squared_error(y_test, lr_pred)),
             np.sqrt(mean_squared_error(y_test, dt_pred))],
    "R2 Score": [r2_score(y_test, lr_pred), r2_score(y_test, dt_pred)],
}).round(4)

print("\nModel comparison:\n", results.to_string(index=False))
results.to_csv("model_results.csv", index=False)

# ---------------------------------------------------------------
# 5. Graphs
# ---------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(df["salary"], bins=20, color="skyblue", edgecolor="black")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig("salary_distribution.png", dpi=120)
plt.close()

plt.figure(figsize=(6, 4))
plt.scatter(df["experience_years"], df["salary"], alpha=0.6, color="green")
plt.title("Experience vs Salary")
plt.xlabel("Experience (years)")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig("experience_vs_salary.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, pred, name in zip(axes, [lr_pred, dt_pred], ["Linear Regression", "Decision Tree"]):
    ax.scatter(y_test, pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_title(name)
    ax.set_xlabel("Actual Salary")
axes[0].set_ylabel("Predicted Salary")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=120)
plt.close()

print("\nDone. Graphs and CSV files saved.")

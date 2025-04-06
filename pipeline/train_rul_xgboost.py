"""
🧠 What This Script Actually Does:
Trains an XGBoost model for RUL prediction

Uses the same feature set and logic as the Random Forest baseline

Selects only the last cycle per engine in the test set for prediction

Evaluates model performance using RMSE for clean comparison

This script helps assess whether XGBoost offers performance gains over the baseline
"""

import joblib
import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Load processed datasets
df_train = pd.read_csv("dataset/df_train_selected.csv")
df_test = pd.read_csv("dataset/df_test_selected.csv")

# Define features (exclude ID, time, label columns)
exclude_cols = ['unit', 'cycle', 'RUL']
feature_cols = [col for col in df_train.columns if col not in exclude_cols]
feature_cols = [col for col in feature_cols if col in df_test.columns]

# Prepare training data
X_train = df_train[feature_cols]
y_train = df_train["RUL"]

# Prepare test data (last cycle per unit only)
df_last_cycle = df_test.groupby("unit").agg({"cycle": "max"}).reset_index()
df_last_cycle = df_test.merge(df_last_cycle, on=["unit", "cycle"], how="inner")


# Load true RUL values
rul_truth = pd.read_csv("dataset/PM_truth.txt", header=None)
rul_truth.columns = ["RUL"]
df_last_cycle["RUL_true"] = rul_truth["RUL"]

X_test = df_last_cycle[feature_cols]
y_test = df_last_cycle["RUL_true"]

# Train XGBoost Regressor
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("📊 XGBoost RMSE:", rmse)



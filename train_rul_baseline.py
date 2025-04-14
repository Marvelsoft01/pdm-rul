"""
What This Script Actually Does:
Trains a baseline Random Forest model for RUL prediction

Handles dynamic feature overlap between train and test sets

Aligns test data with PM_truth.txt using last cycle per engine

Predicts RUL and evaluates using RMSE metric

This is a first-pass benchmark model — useful for sanity check, speed, and feature sensitivity analysis.

"""

import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Load processed training and testing data
df_train = pd.read_csv("dataset/df_train_selected.csv")
df_test = pd.read_csv("dataset/df_test_selected.csv")

# Define feature columns (dynamically)
exclude_cols = ['unit', 'cycle', 'RUL']
feature_cols = [col for col in df_train.columns if col not in exclude_cols]

# Handle feature mismatch between train and test
feature_cols = [col for col in feature_cols if col in df_test.columns]

# Prepare training data
X_train = df_train[feature_cols]
y_train = df_train['RUL']

# Prepare test data (only final cycle of each engine)
df_last_cycle = df_test.groupby("unit").agg({"cycle": "max"}).reset_index()
df_last_cycle = df_test.merge(df_last_cycle, on=["unit", "cycle"], how="inner")

# Load true RUL values
rul_truth = pd.read_csv("dataset/PM_truth.txt", header=None)
rul_truth.columns = ["RUL"]
df_last_cycle["RUL_true"] = rul_truth["RUL"]

X_test = df_last_cycle[feature_cols]
y_test = df_last_cycle["RUL_true"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ RMSE:", rmse)



# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

# Save trained model
joblib.dump(model, "outputs/rf_rul_model.joblib")
print("💾 Model saved at: outputs/rf_rul_model.joblib")

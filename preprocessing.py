"""
preprocessing.py

This script handles loading and preprocessing of the CMAPSS dataset used for Remaining Useful Life (RUL) prediction.

What it does:
- Loads raw training, test, and RUL truth files from the dataset folder.
- Removes unnecessary empty columns caused by inconsistent spacing.
- Assigns clear column names: unit ID, cycle number, and 24 sensor measurements.
- Computes the Remaining Useful Life (RUL) for each row in the training set.
- Converts the RUL truth values for the test set into a DataFrame with the correct structure.
- Saves a reusable version of the processed training data to disk for downstream use.
- Performs sanity checks with `.head()`, `.columns`, and `.shape` to verify data loading and structure.

This script is the foundation of all modeling steps and must be run before feature selection or model training.
"""

import pandas as pd
import os

# Define file paths
dataset_folder = "dataset"  # Adjust this path if needed
train_file = os.path.join(dataset_folder, "PM_train.txt")
test_file = os.path.join(dataset_folder, "PM_test.txt")
truth_file = os.path.join(dataset_folder, "PM_truth.txt")  # Contains actual RUL values for test set

# Load datasets
df_train = pd.read_csv(train_file, sep=" ", header=None, engine="python")
df_test = pd.read_csv(test_file, sep=" ", header=None, engine="python")
df_truth = pd.read_csv(truth_file, sep=" ", header=None, engine="python").iloc[:, 0]

# Remove extra empty columns
df_train.dropna(axis=1, how="all", inplace=True)
df_test.dropna(axis=1, how="all", inplace=True)

# Rename columns (1 unit ID + 1 cycle + 22 sensors)
column_names = ["unit", "cycle"] + [f"sensor_{i}" for i in range(1, 24 + 1)]
df_train.columns = column_names
df_test.columns = column_names
df_truth = df_truth.to_frame(name="RUL")

# --- Compute RUL for Training Data ---
rul_per_unit = df_train.groupby("unit")["cycle"].max().reset_index()
rul_per_unit.columns = ["unit", "max_cycle"]
df_train = df_train.merge(rul_per_unit, on="unit", how="left")
df_train["RUL"] = df_train["max_cycle"] - df_train["cycle"]
df_train.drop(columns=["max_cycle"], inplace=True)

# (Optional) Save preprocessed training set for reuse
df_train.to_csv(os.path.join(dataset_folder, "df_train_with_rul.csv"), index=False)

# --- Output Checks ---
print("✅ Train dataset (with RUL):")
print(df_train.head())
print(df_train.columns)
print(df_train.shape)

print("\n✅ Test dataset:")
print(df_test.head())
print(df_test.columns)
print(df_test.shape)

print("\n✅ Truth RUL values:")
print(df_truth.head())
print(df_truth.columns)
print(df_truth.shape)

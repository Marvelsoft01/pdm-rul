import pandas as pd
from sklearn.model_selection import train_test_split

# Load datasets
df_train = pd.read_csv("dataset/df_train_selected.csv")
df_test = pd.read_csv("dataset/df_test_selected.csv")

# Sanity check: preview column names
print("✅ df_train columns:", df_train.columns.tolist())
print("✅ df_test columns:", df_test.columns.tolist())

# Define consistent feature columns
exclude_cols = ['unit', 'cycle', 'RUL']
feature_cols = [col for col in df_train.columns if col not in exclude_cols]

# Cross-validate features in test set
missing_in_test = [col for col in feature_cols if col not in df_test.columns]
if missing_in_test:
    print(f"⚠️ These features are missing in df_test and will be excluded: {missing_in_test}")
    feature_cols = [col for col in feature_cols if col in df_test.columns]

# Final dataset split
X_train = df_train[feature_cols]
y_train = df_train['RUL']
X_test = df_test[feature_cols]  # No y_test here — use PM_truth.txt separately

# Output shape confirmation
print("🚀 X_train shape:", X_train.shape)
print("🚀 X_test shape:", X_test.shape)

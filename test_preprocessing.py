"""
test_preprocessing.py

Purpose:
Refines feature selection by training a Random Forest on the top 10 correlated features 
(from df_train_selected.csv) and selecting the top 5 based on model-driven importance.

Workflow:
- Loads the dataset with top 10 features previously selected
- Trains a Random Forest Regressor on these features
- Extracts the top 5 most important sensor features from the model
- Combines them with 'unit' and 'cycle' to form the final feature set
- Saves a new dataset containing only the refined top 5 features

Output:
- dataset/df_test_selected.csv — final dataset for model training and prediction
"""


import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load your previously selected 10-feature training dataset
df_train = pd.read_csv("dataset/df_train_selected.csv")

# Separate features and target
exclude_cols = ['unit', 'cycle', 'RUL']
X = df_train.drop(columns=exclude_cols)
y = df_train['RUL']

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
top_5 = importances.sort_values(ascending=False).head(5)

# Final selected features
final_features = ['unit', 'cycle'] + top_5.index.tolist()
print("🔥 Top 5 refined features based on model importance:", top_5.index.tolist())

# Create new dataset
df_train_refined = df_train[final_features + ['RUL']]
df_train_refined.to_csv("dataset/df_test_selected.csv", index=False)
print("✅ Refined df_train_selected.csv saved with top 5 model-informed features.")

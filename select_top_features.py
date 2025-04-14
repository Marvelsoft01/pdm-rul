"""
select_top_features.py

Purpose:
This script performs feature selection from the preprocessed training dataset 
by identifying the top N sensor features that have the strongest correlation with RUL.

Workflow:
- Loads preprocessed training data (with computed RUL) from preprocessing.py
- Computes Pearson correlation of each sensor with RUL
- Selects the top N features based on absolute correlation values
- Creates a refined training dataset including only: unit, cycle, top N sensors, and RUL
- Saves the new dataset for model training

Output:
- dataset/df_train_selected.csv — streamlined dataset for model training
"""



# to get model training data 

from preprocessing import df_train  # Ensure df_train is available
import pandas as pd

# Recalculate correlation with RUL
correlations = df_train.drop(columns=['unit', 'cycle']).corr()
rul_corr = correlations['RUL'].drop('RUL').sort_values(key=abs, ascending=False)

# Select top N features (easier to modify later)
N = 10  # Define the number of features to select
top_features = rul_corr.head(N).index.tolist()
print(f"🔹 Top {N} sensor features selected:", top_features)

# Define the final dataset
features = ['unit', 'cycle'] + top_features
df_train_selected = df_train[features + ['RUL']]

# Save the selected dataset
save_path = "dataset/df_train_selected.csv"
df_train_selected.to_csv(save_path, index=False)
print(f"✅ Feature selection completed. Saved selected dataset to: {save_path}")

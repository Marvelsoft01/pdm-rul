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

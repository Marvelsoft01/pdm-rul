import pandas as pd
import os
from preprocessing import df_test, df_truth  # Make sure df_test and df_truth are available

# Define dataset folder
dataset_folder = "dataset"

# Load selected top features manually or programmatically
top_features = ['sensor_1', 'sensor_5', 'sensor_9', 'sensor_15', 'sensor_20']  # Replace with your actual top 10

# Construct the final list of columns to retain
features = ['unit', 'cycle'] + top_features

# Subset test data
df_test_selected = df_test[features]

# Attach RUL values (same order as test units)
df_test_selected['RUL'] = df_truth

# Save processed file
output_path = os.path.join(dataset_folder, "df_test_selected.csv")
df_test_selected.to_csv(output_path, index=False)

print("✅ Test dataset processing completed. File saved as df_test_selected.csv.")

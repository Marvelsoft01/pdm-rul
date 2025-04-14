"""
used to display features the model trainned on
"""

import joblib

# Load your trained model
model = joblib.load("outputs/rf_rul_model.joblib")

# Print the features it expects
print("Model was trained on these features:")
print(model.feature_names_in_)





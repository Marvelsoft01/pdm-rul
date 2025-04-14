"""
predict.py

This script launches a Streamlit-based user interface for interacting with the trained Random Forest model.

What it does:
- Loads a previously trained RUL (Remaining Useful Life) prediction model from the local file system.
- Provides two ways to input sensor readings: file upload (CSV) or manual entry via the UI.
- Predicts the RUL based on the input values.
- Offers technician-friendly feedback depending on the predicted RUL severity.
- Includes downloadable sample CSV for easy testing.

This interface is ideal for local demos, pilot testing, or as a starting point for future production deployment.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import gzip


with gzip.open("outputs/rf_rul_model.joblib.gz", "rb") as f:
    model = joblib.load(f)
# Load trained model
MODEL_PATH = "outputs/rf_rul_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please train and save it as 'rf_rul_model.joblib'.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Streamlit UI
st.title("Predict Remaining Useful Life (RUL)")
st.markdown("Upload a CSV file with sensor readings or enter them manually to predict RUL.")

# Upload input file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        input_df = input_df[model.feature_names_in_]
        prediction = model.predict(input_df)
        st.success(f"Predicted RULs: {prediction.round(2).tolist()}")

    except KeyError as e:
        st.error(f"Uploaded CSV is missing required columns: {e}")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Sample input for download
sample_data = pd.DataFrame([{
    "sensor_14": 10.5,
    "sensor_7": 0.5,
    "sensor_18": 100.0,
    "sensor_24": 20.0,
    "sensor_6": 515.0
}])

st.markdown("### Need a sample file?")
st.download_button(
    label="Download Sample CSV",
    data=sample_data.to_csv(index=False),
    file_name="sample_sensor_input.csv",
    mime="text/csv"
)

# Manual entry
st.markdown("### Or enter values manually:")
manual_input = {}
for feature in model.feature_names_in_:
    manual_input[feature] = st.number_input(f"{feature}", step=0.1)

if st.button("Predict RUL"):
    input_df = pd.DataFrame([manual_input])
    input_df = input_df.astype(float)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0.0)
    prediction = model.predict(input_df)[0]
    rounded_rul = round(prediction, 2)

    # RUL display
    st.success(f"Predicted RUL: {rounded_rul} cycles")

    # Interpretation
    if rounded_rul < 30:
        st.warning("Urgent: Component is nearing failure. Schedule maintenance immediately.")
    elif rounded_rul < 80:
        st.info("Moderate wear: Plan preventive maintenance soon.")
    else:
        st.success("Component is in healthy range. No immediate action required.")

    # Explanation block
    st.markdown("""
    ---
    ### What does this number mean?
    - RUL (Remaining Useful Life) estimates how many more cycles the component can complete before likely failure.
    - 1 cycle = one full operation loop (e.g., shift, hour, flight).
    - Use this value to guide preventive maintenance scheduling.
    """)

st.markdown("---")
st.caption("Model trained using Random Forest with RMSE ≈ 43.57 cycles. Built for technician insight and action.")
"""
predict.py

This script launches a Streamlit-based user interface for interacting with the trained Random Forest model.

What it does:
- Loads a previously trained RUL (Remaining Useful Life) prediction model from the local file system.
- Provides two ways to input sensor readings: file upload (CSV) or manual entry via the UI.
- Predicts the RUL based on the input values.
- Offers technician-friendly feedback depending on the predicted RUL severity.
- Includes downloadable sample CSV for easy testing.

This interface is ideal for local demos, pilot testing, or as a starting point for future production deployment.
"""

import streamlit as st
import pandas as pd
import joblib
import gzip
import os

# Load trained model (compressed .gz file)
MODEL_PATH = "outputs/rf_rul_model.joblib.gz"

if not os.path.exists(MODEL_PATH):
    st.error(f"Compressed model file not found at: {MODEL_PATH}")
    st.stop()

with gzip.open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)

# Streamlit UI
st.title("Predict Remaining Useful Life (RUL)")
st.markdown("Upload a CSV file with sensor readings or enter them manually to predict RUL.")

# Upload input file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        input_df = input_df[model.feature_names_in_]
        prediction = model.predict(input_df)
        st.success(f"Predicted RULs: {prediction.round(2).tolist()}")

    except KeyError as e:
        st.error(f"Uploaded CSV is missing required columns: {e}")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Sample input for download
sample_data = pd.DataFrame([{
    "sensor_14": 10.5,
    "sensor_7": 0.5,
    "sensor_18": 100.0,
    "sensor_24": 20.0,
    "sensor_6": 515.0
}])

st.markdown("### Need a sample file?")
st.download_button(
    label="Download Sample CSV",
    data=sample_data.to_csv(index=False),
    file_name="sample_sensor_input.csv",
    mime="text/csv"

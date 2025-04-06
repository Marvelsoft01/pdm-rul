"""
🧠 What This App Actually Does:
Loads the trained Random Forest RUL model

Provides a Streamlit interface for users to:
– Upload sensor data OR enter values manually
– Predict Remaining Useful Life (RUL)
– View result and input confirmation

Bridges the ML model into a product experience for CMMS founders, engineers, and users.
"""
import io
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("outputs/rf_rul_model.joblib")

# Define expected feature columns (same as used during training)
expected_features = ['sensor_5', 'sensor_15', 'sensor_20']

st.title("🔧 Predict Remaining Useful Life (RUL)")
st.markdown("Upload a CSV file with sensor readings or enter them manually to predict RUL.")






# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        if all(f in input_df.columns for f in expected_features):
            prediction = model.predict(input_df[expected_features])
            st.success(f"📈 Predicted RULs: {prediction.round(2).tolist()}")
        else:
            st.warning(f"Missing required features. Expected columns: {expected_features}")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Sample data for download
sample_data = pd.DataFrame([{
    "sensor_5": 490.2,
    "sensor_15": 8.75,
    "sensor_20": 38.9
}, {
    "sensor_5": 512.6,
    "sensor_15": 9.12,
    "sensor_20": 39.5
}])

st.markdown("### 🧾 Need a sample file?")
st.download_button(
    label="📥 Download Sample CSV",
    data=sample_data.to_csv(index=False),
    file_name="sample_sensor_input.csv",
    mime="text/csv"
)

# Manual input
st.markdown("### Or enter values manually:")
manual_input = {}
for feature in expected_features:
    manual_input[feature] = st.number_input(f"{feature}", step=0.1)

# if st.button("🔍 Predict RUL"):
#     input_df = pd.DataFrame([manual_input])
#     prediction = model.predict(input_df)[0]
#     st.success(f"⚙️ Predicted RUL: **{round(prediction, 2)} cycles**")

if st.button("🔍 Predict RUL"):
    input_df = pd.DataFrame([manual_input])

    # 🔒 Ensure input columns match exactly what model was trained on
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0.0)
    
    # 🧠 Optional: Force float
    input_df = input_df.astype(float)

    # 🔮 Predict
    prediction = model.predict(input_df)[0]
    st.success(f"🔧 Predicted RUL: **{round(prediction, 2)} cycles**")



st.markdown("---")
st.caption("Model trained using Random Forest with RMSE of ~43.57 cycles.")




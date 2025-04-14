"""
predict.py

This script launches a Streamlit-based UI to predict Remaining Useful Life (RUL).
If the model isn't trained yet, it automatically runs the full data pipeline:
1. preprocessing.py
2. select_top_features.py
3. test_preprocessing.py
4. train_rul_baseline.py

Perfect for one-click local demos, testing, or onboarding non-technical users.
"""

import streamlit as st
import pandas as pd
import joblib
import os
import subprocess

# ----------------------------
# Step 1: Ensure model exists
# ----------------------------
MODEL_PATH = "outputs/rf_rul_model.joblib"

if not os.path.exists(MODEL_PATH):
    st.warning("Model not found. Running full pipeline to generate model...")

    try:
        subprocess.run(["python", "preprocessing.py"], check=True)
        subprocess.run(["python", "select_top_features.py"], check=True)
        subprocess.run(["python", "test_preprocessing.py"], check=True)
        subprocess.run(["python", "train_rul_baseline.py"], check=True)
        st.success("✅ Model pipeline completed successfully!")

    except subprocess.CalledProcessError as e:
        st.error(f"❌ Pipeline failed: {e}")
        st.stop()

# ----------------------------
# Step 2: Load Trained Model
# ----------------------------
model = joblib.load(MODEL_PATH)

# ----------------------------
# Step 3: UI — Streamlit Frontend
# ----------------------------
st.title("⚙️ Predict Remaining Useful Life (RUL)")
st.markdown("Upload a CSV file with sensor readings or enter them manually to predict RUL.")

# Upload input file
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        input_df = input_df[model.feature_names_in_]
        prediction = model.predict(input_df)
        st.success(f"📈 Predicted RULs: {prediction.round(2).tolist()}")
    except KeyError as e:
        st.error(f"❌ Missing required columns: {e}")
    except Exception as e:
        st.error(f"❌ File read error: {e}")

# Sample CSV download
sample_data = pd.DataFrame([{
    "sensor_14": 10.5,
    "sensor_7": 0.5,
    "sensor_18": 100.0,
    "sensor_24": 20.0,
    "sensor_6": 515.0
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
for feature in model.feature_names_in_:
    manual_input[feature] = st.number_input(f"{feature}", step=0.1)

if st.button("🔍 Predict RUL"):
    input_df = pd.DataFrame([manual_input])
    input_df = input_df.astype(float)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0.0)

    prediction = model.predict(input_df)[0]
    rounded_rul = round(prediction, 2)

    st.success(f"🔧 Predicted RUL: **{rounded_rul} cycles**")

    if rounded_rul < 30:
        st.warning("⚠️ Urgent: Component is nearing failure. Schedule maintenance immediately.")
    elif rounded_rul < 80:
        st.info("🛠️ Moderate wear: Plan preventive maintenance soon.")
    else:
        st.success("✅ Component is in healthy range. No immediate action required.")

    st.markdown("""
    ---
    ### 📘 What does this number mean?
    - **RUL** = how many more cycles this component can likely survive before failure.
    - 1 cycle = one full operation run (e.g., a shift, a flight, a mission).
    - Use this to optimize preventive maintenance planning.
    """)

st.markdown("---")
st.caption("Model trained with Random Forest. Auto-runs if model file is missing.")

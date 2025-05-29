import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Title
st.set_page_config(page_title="Zimbabwe Vehicle Insurance Fraud Detector")
st.title("🚗 Zimbabwe Vehicle Insurance Fraud Detection App")
st.write("Predict the probability of a vehicle insurance fraud case using your model.")

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load("models/zimbabwe_vehicle_fraud_model_final.pkl")  # Path inside repo
    return model

model = load_model()

# Input fields for prediction
st.header("Enter Claim Information")

driver_age = st.number_input("Driver Age", min_value=18, max_value=100, value=30)
driver_gender = st.selectbox("Driver Gender", ["Male", "Female"])
accident_location = st.text_input("Accident Location", "Harare")
vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck", "Bus", "Other"])
vehicle_model = st.text_input("Vehicle Model", "Toyota Fortuner")
vehicle_year = st.number_input("Vehicle Year", min_value=1980, max_value=2025, value=2018)
vehicle_value = st.number_input("Vehicle Value (USD)", min_value=500, max_value=50000, value=15000)
claim_type = st.selectbox("Claim Type", ["Theft", "Accident", "Fire", "Flood", "Other"])
days_active = st.number_input("Days Policy Active", min_value=1, max_value=5000, value=365)
previous_claims = st.number_input("Number of Previous Claims", min_value=0, max_value=10, value=1)
police_report = st.selectbox("Police Report Filed?", ["Yes", "No"])

# Convert police report to binary
police_report_binary = 1 if police_report == "Yes" else 0

# Submit button
if st.button("Predict Fraud Probability"):
    try:
        # Input data as DataFrame (you may need to one-hot encode or label encode in real use)
        input_data = pd.DataFrame([{
            'driver_age': driver_age,
            'driver_gender': driver_gender,
            'accident_location': accident_location,
            'vehicle_type': vehicle_type,
            'model': vehicle_model,
            'year': vehicle_year,
            'value': vehicle_value,
            'claim_type': claim_type,
            'days_active': days_active,
            'previous_claims': previous_claims,
            'police_report': police_report_binary
        }])

        # Ensure your model pipeline includes preprocessing
        fraud_proba = model.predict_proba(input_data)[0][1]
        risk_level = "High" if fraud_proba >= 0.7 else "Medium" if fraud_proba >= 0.4 else "Low"

        st.subheader("🧾 Prediction Result")
        st.metric(label="Fraud Probability", value=f"{fraud_proba:.2%}")
        st.metric(label="Risk Level", value=risk_level)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

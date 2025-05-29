import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('zimbabwe_vehicle_fraud_model_final.pkl')

# Function to make predictions
def predict_fraud(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit app layout
st.title("Vehicle Insurance Fraud Detection")
st.write("Enter the details below:")

# Input fields for user data
driver_age = st.number_input("Driver Age:", min_value=18, max_value=100)
vehicle_type_freq = st.number_input("Vehicle Type Frequency:")
model_freq = st.number_input("Model Frequency:")
accident_location_freq = st.number_input("Accident Location Frequency:")
claim_type_freq = st.number_input("Claim Type Frequency:")
year = st.number_input("Year of Vehicle:")
value = st.number_input("Value of Vehicle:")
days_active = st.number_input("Days Active:")
previous_claims = st.number_input("Previous Claims:")
police_report = st.number_input("Police Report (0 or 1):")

# Prepare input data for the model
input_data = pd.DataFrame({
    'driver_age': [driver_age],
    'vehicle_type_freq': [vehicle_type_freq],
    'model_freq': [model_freq],
    'accident_location_freq': [accident_location_freq],
    'claim_type_freq': [claim_type_freq],
    'year': [year],
    'value': [value],
    'days_active': [days_active],
    'previous_claims': [previous_claims],
    'police_report': [police_report]
})

if st.button("Predict"):
    prediction = predict_fraud(input_data)
    if prediction[0] == 1:
        st.success("Fraud Detected!")
    else:
        st.success("No Fraud Detected.")

# Run the app
if __name__ == "__main__":
    st.run()

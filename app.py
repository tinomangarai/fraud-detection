import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the saved model
model = joblib.load('models/zimbabwe_vehicle_fraud_model_final.pkl')

# Preprocessing function (same as training)
def preprocess_input(input_data):
    # Frequency encoding mappings (replace with your actual mappings)
    freq_mappings = {
        'make': {'Toyota': 0.3, 'Nissan': 0.2, 'Isuzu': 0.1, 'Honda': 0.05},
        'model': {'Hilux': 0.25, 'D-Max': 0.15, 'Navara': 0.1},
        'vehicle_type': {'Commercial': 0.4, 'Minibus': 0.3, 'Truck': 0.2},
        'claim_type': {'Collision': 0.5, 'Glass Breakage': 0.3, 'Theft': 0.2},
        'driver_gender': {'M': 0.6, 'F': 0.4},
        'accident_location': {'Gweru': 0.2, 'Mutare': 0.15, 'Unknown': 0.1},
        'province': {'Midlands': 0.25, 'Manicaland': 0.2, 'Mashonaland West': 0.15},
        'city': {'Gweru': 0.2, 'Mutare': 0.15, 'Chinhoyi': 0.1}
    }
    
    # Apply frequency encoding
    for col in ['make', 'model', 'vehicle_type', 'claim_type', 'driver_gender', 'accident_location', 'province', 'city']:
        input_data[f'{col}_freq'] = input_data[col].map(freq_mappings.get(col, {}))
    
    # Log transform for 'amount'
    input_data['log_amount'] = np.log1p(input_data['amount'])
    
    # Interaction features
    input_data['age_amount_interaction'] = input_data['driver_age'] * input_data['log_amount']
    input_data['prev_claims_days_active'] = input_data['previous_claims'] * input_data['days_active']
    
    # Binning age (must match training)
    input_data['age_bin'] = pd.cut(input_data['driver_age'], bins=[0, 18, 30, 45, 60, 100], 
                                 labels=['0-18', '19-30', '31-45', '46-60', '60+'])
    input_data = pd.get_dummies(input_data, columns=['age_bin'], prefix='age')
    
    # Ensure all expected columns are present
    expected_cols = [
        "driver_age", "driver_gender_freq", "accident_location_freq", 
        "vehicle_type_freq", "model_freq", "year", "value", 
        "claim_type_freq", "days_active", "previous_claims", "police_report",
        "log_amount", "age_amount_interaction", "prev_claims_days_active",
        "age_0-18", "age_19-30", "age_31-45", "age_46-60", "age_60+"
    ]
    for col in expected_cols:
        if col not in input_data:
            input_data[col] = 0
    
    return input_data[expected_cols]

# Streamlit UI
st.title("🇿🇼 Zimbabwe Vehicle Insurance Fraud Detection")

# Input form
with st.form("fraud_form"):
    st.header("Claim Details")
    
    col1, col2 = st.columns(2)
    with col1:
        province = st.selectbox("Province", ["Midlands", "Manicaland", "Mashonaland West", "Masvingo"])
        city = st.text_input("City", "Gweru")
        make = st.selectbox("Vehicle Make", ["Toyota", "Nissan", "Isuzu", "Honda"])
        model = st.text_input("Vehicle Model", "Hilux")
        vehicle_type = st.selectbox("Vehicle Type", ["Commercial", "Minibus", "Truck"])
        year = st.number_input("Vehicle Year", min_value=1980, max_value=2024, value=2010)
        value = st.number_input("Vehicle Value ($)", min_value=0, value=39046)
    
    with col2:
        claim_type = st.selectbox("Claim Type", ["Collision", "Glass Breakage", "Theft", "Third Party"])
        amount = st.number_input("Claim Amount ($)", min_value=0, value=25128)
        driver_age = st.number_input("Driver Age", min_value=18, max_value=100, value=31)
        driver_gender = st.selectbox("Driver Gender", ["M", "F"])
        accident_location = st.text_input("Accident Location", "Gweru")
        police_report = st.selectbox("Police Report Filed?", [1, 0])
        previous_claims = st.number_input("Previous Claims", min_value=0, value=0)
        days_active = st.number_input("Days Active (Insurance)", min_value=0, value=1611)
    
    submitted = st.form_submit_button("Check Fraud Risk")

# Prediction
if submitted:
    input_data = pd.DataFrame({
        'province': [province],
        'city': [city],
        'make': [make],
        'model': [model],
        'vehicle_type': [vehicle_type],
        'year': [year],
        'value': [value],
        'claim_type': [claim_type],
        'amount': [amount],
        'driver_age': [driver_age],
        'driver_gender': [driver_gender],
        'accident_location': [accident_location],
        'police_report': [police_report],
        'previous_claims': [previous_claims],
        'days_active': [days_active]
    })
    
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    proba = model.predict_proba(processed_data)[0]
    
    if prediction[0] == 1:
        st.error(f"🚨 **High Fraud Risk** ({proba[1]*100:.2f}% probability)")
    else:
        st.success(f"✅ **Low Fraud Risk** ({proba[0]*100:.2f}% probability)")

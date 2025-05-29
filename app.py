import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import pairwise_distances

# Paths
MODEL_PATH = "models/zimbabwe_vehicle_fraud_model_final.pkl"
DATA_PATH = "zimbabwe_vehicle_insurance_fraud.csv"

# Load model and reference data
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    freq_maps = {}
    for col in ['driver_gender', 'accident_location', 'vehicle_type', 'model', 'claim_type']:
        freq_maps[col] = data[col].value_counts(normalize=True).to_dict()

    return model, data, freq_maps

model, reference_data, freq_maps = load_resources()

# Features used by the model
selected_features = [
    "driver_age", "driver_gender_freq", "accident_location_freq",
    "vehicle_type_freq", "model_freq", "year", "value",
    "claim_type_freq", "days_active", "previous_claims", "police_report"
]

# Input form
st.title("🚗 Zimbabwe Vehicle Insurance Fraud Detection App")
st.markdown("Predict the probability of a vehicle insurance fraud case using your model.")

with st.form("fraud_form"):
    driver_age = st.number_input("Driver Age", min_value=16, max_value=100, value=30)
    driver_gender = st.selectbox("Driver Gender", ['Male', 'Female'])
    accident_location = st.text_input("Accident Location", "Harare")
    vehicle_type = st.selectbox("Vehicle Type", ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Other'])
    model_input = st.text_input("Vehicle Model", "Toyota")
    year = st.number_input("Vehicle Year", min_value=1980, max_value=2025, value=2015)
    value = st.number_input("Vehicle Value (USD)", min_value=100.0, max_value=100000.0, value=5000.0)
    claim_type = st.selectbox("Claim Type", ['Collision', 'Theft', 'Fire', 'Flood', 'Other'])
    days_active = st.number_input("Days Active", min_value=0, max_value=10000, value=365)
    previous_claims = st.number_input("Previous Claims", min_value=0, max_value=50, value=1)
    police_report = st.selectbox("Police Report Filed", ['Yes', 'No'])

    submitted = st.form_submit_button("Predict Fraud")

def preprocess_input(data):
    df = pd.DataFrame([data])

    df['driver_gender_freq'] = df['driver_gender'].map(freq_maps['driver_gender']).fillna(0)
    df['accident_location_freq'] = df['accident_location'].map(freq_maps['accident_location']).fillna(0)
    df['vehicle_type_freq'] = df['vehicle_type'].map(freq_maps['vehicle_type']).fillna(0)
    df['model_freq'] = df['model'].map(freq_maps['model']).fillna(0)
    df['claim_type_freq'] = df['claim_type'].map(freq_maps['claim_type']).fillna(0)

    df['police_report'] = 1 if df['police_report'].iloc[0].lower() == 'yes' else 0

    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0

    return df[selected_features]

def find_similar_cases(user_input, n=5):
    try:
        user_processed = preprocess_input(user_input)

        ref = reference_data.copy()
        for col in ['driver_gender', 'accident_location', 'vehicle_type', 'model', 'claim_type']:
            ref[f'{col}_freq'] = ref[col].map(freq_maps[col]).fillna(0)

        ref_processed = ref[selected_features].fillna(0)
        similarity = 1 - pairwise_distances(user_processed, ref_processed, metric='cosine')[0]
        top_indices = np.argsort(similarity)[-n:][::-1]

        top_cases = ref.iloc[top_indices].copy()
        top_cases['similarity'] = similarity[top_indices]

        return top_cases[[
            'claim_id', 'province', 'city', 'make', 'model', 'vehicle_type', 'year',
            'value', 'claim_type', 'amount', 'driver_age', 'driver_gender',
            'accident_location', 'police_report', 'previous_claims', 'days_active',
            'is_fraud', 'similarity'
        ]]
    except Exception as e:
        st.error(f"Error finding similar cases: {e}")
        return pd.DataFrame()

if submitted:
    user_input = {
        "driver_age": driver_age,
        "driver_gender": driver_gender,
        "accident_location": accident_location,
        "vehicle_type": vehicle_type,
        "model": model_input,
        "year": year,
        "value": value,
        "claim_type": claim_type,
        "days_active": days_active,
        "previous_claims": previous_claims,
        "police_report": police_report
    }

    try:
        processed = preprocess_input(user_input)
        prob = model.predict_proba(processed)[0][1]
        fraud_percent = round(prob * 100, 2)

        similar_cases = find_similar_cases(user_input)
        if not similar_cases.empty:
            similar_fraud_rate = similar_cases['is_fraud'].mean()
            if similar_fraud_rate > 0.5 and prob < 0.5:
                prob = min(prob + 0.3, 0.99)
                fraud_percent = round(prob * 100, 2)

        if prob >= 0.7:
            risk_level = "🔴 High Risk"
        elif prob >= 0.4:
            risk_level = "🟠 Medium Risk"
        else:
            risk_level = "🟢 Low Risk"

        st.success(f"**Fraud Probability: {fraud_percent}%**")
        st.markdown(f"### Risk Assessment: {risk_level}")

        if not similar_cases.empty:
            st.markdown("### 🔍 Similar Historical Cases")
            st.dataframe(similar_cases)

    except Exception as e:
        st.error(f"Prediction error: {e}")

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import pairwise_distances

# Configuration
MODEL_PATH = 'models/zimbabwe_vehicle_fraud_model_final.pkl'
DATA_PATH = 'zimbabwe_vehicle_insurance_fraud.csv'

class VehicleFraudDetector:
    def __init__(self):
        self.model = None
        self.reference_set = None
        self.freq_maps = {}
        self.selected_features = [
            "driver_age", "driver_gender_freq", "accident_location_freq",
            "vehicle_type_freq", "model_freq", "year", "value",
            "claim_type_freq", "days_active", "previous_claims", "police_report"
        ]
        self._load_resources()

    def _load_resources(self):
        """Load model and training data"""
        self.model = joblib.load(MODEL_PATH)
        full_data = pd.read_csv(DATA_PATH)
        self.reference_set = full_data.copy()

        # Create frequency maps for categorical variables
        for col in ['driver_gender', 'accident_location', 'vehicle_type', 'model', 'claim_type']:
            self.freq_maps[col] = full_data[col].value_counts(normalize=True).to_dict()

    def preprocess_input(self, input_data):
        """Transform input data into model-ready features"""
        data = pd.DataFrame([input_data])

        # Process numerical fields
        num_cols = ['driver_age', 'year', 'value', 'days_active', 'previous_claims', 'police_report']
        for col in num_cols:
            data[col] = pd.to_numeric(data.get(col, 0), errors='coerce').fillna(0)

        # Calculate frequency features
        data['driver_gender_freq'] = data['driver_gender'].map(self.freq_maps['driver_gender']).fillna(0)
        data['accident_location_freq'] = data['accident_location'].map(self.freq_maps['accident_location']).fillna(0)
        data['vehicle_type_freq'] = data['vehicle_type'].map(self.freq_maps['vehicle_type']).fillna(0)
        data['model_freq'] = data['model'].map(self.freq_maps['model']).fillna(0)
        data['claim_type_freq'] = data['claim_type'].map(self.freq_maps['claim_type']).fillna(0)

        # Ensure all selected features exist
        for feature in self.selected_features:
            if feature not in data.columns:
                data[feature] = 0
                
        return data[self.selected_features]

    def find_similar_cases(self, input_data, n=5):
        """Find similar historical cases"""
        input_processed = self.preprocess_input(input_data)
        ref_data = self.reference_set.copy()

        # Prepare reference data with same features
        for col in ['driver_gender', 'accident_location', 'vehicle_type', 'model', 'claim_type']:
            ref_data[f'{col}_freq'] = ref_data[col].map(self.freq_maps[col]).fillna(0)

        ref_processed = ref_data[self.selected_features].fillna(0)

        # Calculate similarities
        similarities = 1 - pairwise_distances(input_processed, ref_processed, metric='cosine')[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-n:][::-1]
        similar_cases = ref_data.iloc[top_indices].copy()
        similar_cases['similarity'] = similarities[top_indices]

        return similar_cases

# Initialize detector
fraud_detector = VehicleFraudDetector()

# Streamlit app layout
st.title("Vehicle Insurance Fraud Detection")
st.write("Enter the details below:")

# Input fields for user data
driver_age = st.number_input("Driver Age:", min_value=18, max_value=100)
driver_gender = st.selectbox("Driver Gender:", ["Male", "Female"])
accident_location = st.selectbox("Accident Location:", fraud_detector.freq_maps['accident_location'].keys())
vehicle_type = st.selectbox("Vehicle Type:", fraud_detector.freq_maps['vehicle_type'].keys())
model = st.selectbox("Model:", fraud_detector.freq_maps['model'].keys())
claim_type = st.selectbox("Claim Type:", fraud_detector.freq_maps['claim_type'].keys())
year = st.number_input("Year of Vehicle:")
value = st.number_input("Value of Vehicle:")
days_active = st.number_input("Days Active:")
previous_claims = st.number_input("Previous Claims:")
police_report = st.number_input("Police Report (0 or 1):")

# Collect input data
input_data = {
    'driver_age': driver_age,
    'driver_gender': driver_gender,
    'accident_location': accident_location,
    'vehicle_type': vehicle_type,
    'model': model,
    'claim_type': claim_type,
    'year': year,
    'value': value,
    'days_active': days_active,
    'previous_claims': previous_claims,
    'police_report': police_report,
}

if st.button("Predict"):
    processed_data = fraud_detector.preprocess_input(input_data)
    fraud_prob = fraud_detector.model.predict_proba(processed_data)[0][1]
    fraud_percent = round(fraud_prob * 100, 2)

    # Get similar cases
    similar_cases = fraud_detector.find_similar_cases(input_data)

    # Risk classification
    if fraud_prob >= 0.7:
        risk_level, risk_class = "High Risk", "high-risk"
    elif fraud_prob >= 0.4:
        risk_level, risk_class = "Medium Risk", "medium-risk"
    else:
        risk_level, risk_class = "Low Risk", "low-risk"

    # Display results
    st.success(f"Fraud Probability: {fraud_percent}% - Risk Level: {risk_level}")
    if not similar_cases.empty:
        st.write("Similar Cases:")
        st.dataframe(similar_cases[['claim_id', 'province', 'city', 'make', 'model', 'is_fraud', 'similarity']])

# Run the app
if __name__ == "__main__":
    st.run()

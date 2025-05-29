import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import pairwise_distances

app = Flask(__name__)

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
        try:
            self.model = joblib.load(MODEL_PATH)
            full_data = pd.read_csv(DATA_PATH)
            self.reference_set = full_data.copy()
            
            # Create frequency maps for categorical variables
            for col in ['driver_gender', 'accident_location', 'vehicle_type', 'model', 'claim_type']:
                self.freq_maps[col] = full_data[col].value_counts(normalize=True).to_dict()
                
        except Exception as e:
            raise RuntimeError(f"Failed to load resources: {str(e)}")

    def preprocess_input(self, input_data):
        """Transform raw form data into model-ready features"""
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
        try:
            input_processed = self.preprocess_input(input_data)
            ref_data = self.reference_set.copy()
            
            # Prepare reference data with same features
            for col in ['driver_gender', 'accident_location', 'vehicle_type', 'model', 'claim_type']:
                ref_data[f'{col}_freq'] = ref_data[col].map(self.freq_maps[col]).fillna(0)
            
            ref_processed = ref_data[self.selected_features].fillna(0)
            
            # Calculate similarities
            similarities = 1 - pairwise_distances(
                input_processed, 
                ref_processed, 
                metric='cosine'
            )[0]
            
            # Get top matches
            top_indices = np.argsort(similarities)[-n:][::-1]
            similar_cases = ref_data.iloc[top_indices].copy()
            similar_cases['similarity'] = similarities[top_indices]
            
            return similar_cases[[
                'claim_id', 'province', 'city', 'make', 'model', 
                'vehicle_type', 'year', 'value', 'claim_type', 
                'amount', 'driver_age', 'driver_gender', 
                'accident_location', 'police_report',
                'previous_claims', 'days_active', 'is_fraud', 'similarity'
            ]]
        except Exception as e:
            app.logger.error(f"Error finding similar cases: {str(e)}")
            return pd.DataFrame()

# Initialize detector
try:
    fraud_detector = VehicleFraudDetector()
except Exception as e:
    print(f"Failed to initialize: {str(e)}")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        
        try:
            # Get prediction
            processed_data = fraud_detector.preprocess_input(form_data)
            fraud_prob = fraud_detector.model.predict_proba(processed_data)[0][1]
            fraud_percent = round(fraud_prob * 100, 2)
            
            # Get similar cases
            similar_cases = fraud_detector.find_similar_cases(form_data)
            
            # Adjust prediction
            if not similar_cases.empty:
                similar_fraud_rate = similar_cases['is_fraud'].mean()
                if similar_fraud_rate > 0.5 and fraud_prob < 0.5:
                    fraud_prob = min(fraud_prob + 0.3, 0.99)
                    fraud_percent = round(fraud_prob * 100, 2)
            
            # Risk classification
            if fraud_prob >= 0.7:
                risk_level, risk_class = "High Risk", "high-risk"
            elif fraud_prob >= 0.4:
                risk_level, risk_class = "Medium Risk", "medium-risk"
            else:
                risk_level, risk_class = "Low Risk", "low-risk"
            
            return render_template('result.html',
                fraud_prob=fraud_percent,
                risk_level=risk_level,
                risk_class=risk_class,
                similar_cases=similar_cases.to_dict('records'),
                form_data=form_data)
            
        except Exception as e:
            return render_template('index.html',
                errors=[f"Error: {str(e)}"],
                form_data=form_data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST':
        try:
            input_data = request.get_json()
            processed_data = fraud_detector.preprocess_input(input_data)
            fraud_prob = fraud_detector.model.predict_proba(processed_data)[0][1]
            
            return jsonify({
                'fraud_probability': float(fraud_prob),
                'risk_level': "High" if fraud_prob >= 0.7 else "Medium" if fraud_prob >= 0.4 else "Low",
                'status': 'success'
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

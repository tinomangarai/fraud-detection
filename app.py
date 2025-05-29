import streamlit as st
import pandas as pd
import numpy as np
import joblib
import imblearn
from sklearn.metrics import pairwise_distances

# ----------------------------------------------------------------------------------
# Paths inside the repo
# ----------------------------------------------------------------------------------
MODEL_PATH = "models/zimbabwe_vehicle_fraud_model_final.pkl"
DATA_PATH = "zimbabwe_vehicle_insurance_fraud.csv"

# ----------------------------------------------------------------------------------
# Load model and reference dataset once (cached)
# ----------------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)  # loads Pipeline(SMOTE → GBC)
    ref = pd.read_csv(DATA_PATH)

    freq = {}
    for col in ["driver_gender", "accident_location", "vehicle_type", "model", "claim_type"]:
        freq[col] = ref[col].value_counts(normalize=True).to_dict()

    return model, ref, freq

model, reference_data, freq_maps = load_resources()

# ----------------------------------------------------------------------------------
# Feature list used by the model
# ----------------------------------------------------------------------------------
selected_features = [
    "driver_age", "driver_gender_freq", "accident_location_freq",
    "vehicle_type_freq", "model_freq", "year", "value",
    "claim_type_freq", "days_active", "previous_claims", "police_report",
    "log_amount", "age_amount_interaction", "prev_claims_days_active"
]

# ----------------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------------
def preprocess_input(record: dict) -> pd.DataFrame:
    df = pd.DataFrame([record])

    # Frequency encode categorical variables
    df["driver_gender_freq"] = df["driver_gender"].map(freq_maps["driver_gender"]).fillna(0)
    df["accident_location_freq"] = df["accident_location"].map(freq_maps["accident_location"]).fillna(0)
    df["vehicle_type_freq"] = df["vehicle_type"].map(freq_maps["vehicle_type"]).fillna(0)
    df["model_freq"] = df["model"].map(freq_maps["model"]).fillna(0)
    df["claim_type_freq"] = df["claim_type"].map(freq_maps["claim_type"]).fillna(0)

    # Binary conversion
    df["police_report"] = (df["police_report"].str.lower() == "yes").astype(int)

    # Derived features
    df["log_amount"] = np.log1p(df["amount"])
    df["age_amount_interaction"] = df["driver_age"] * df["log_amount"]
    df["prev_claims_days_active"] = df["previous_claims"] * df["days_active"]

    # Fill missing values for unused features
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    return df[selected_features]

# ----------------------------------------------------------------------------------
# Find similar cases from historical data
# ----------------------------------------------------------------------------------
def find_similar_cases(record, n=5):
    user_vec = preprocess_input(record)

    ref = reference_data.copy()
    for col in ["driver_gender", "accident_location", "vehicle_type", "model", "claim_type"]:
        ref[f"{col}_freq"] = ref[col].map(freq_maps[col]).fillna(0)

    ref["log_amount"] = np.log1p(ref["amount"])
    ref["age_amount_interaction"] = ref["driver_age"] * ref["log_amount"]
    ref["prev_claims_days_active"] = ref["previous_claims"] * ref["days_active"]
    ref["police_report"] = (ref["police_report"].str.lower() == "yes").astype(int)

    similarities = 1 - pairwise_distances(user_vec, ref[selected_features], metric="cosine")[0]
    top_idx = np.argsort(similarities)[-n:][::-1]
    top = ref.iloc[top_idx].copy()
    top["similarity"] = similarities[top_idx]
    return top

# ----------------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Zimbabwe Vehicle Insurance Fraud Detector")
st.title("🚗 Zimbabwe Vehicle Insurance Fraud Detection App")
st.write("Predict the probability that a submitted claim is fraudulent.")

with st.form("claim_form"):
    driver_age = st.number_input("Driver Age", 16, 100, 30)
    driver_gender = st.selectbox("Driver Gender", ["Male", "Female"])
    accident_location = st.text_input("Accident Location", "Harare")
    vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck", "Hatchback", "Other"])
    vehicle_model = st.text_input("Vehicle Model", "Toyota")
    year = st.number_input("Vehicle Year", 1980, 2025, 2018)
    value = st.number_input("Vehicle Value (USD)", 100.0, 100000.0, 5000.0)
    claim_type = st.selectbox("Claim Type", ["Collision", "Theft", "Fire", "Flood", "Other"])
    days_active = st.number_input("Days Policy Active", 0, 10000, 365)
    previous_claims = st.number_input("Previous Claims", 0, 50, 1)
    police_report = st.selectbox("Police Report Filed", ["Yes", "No"])
    amount = st.number_input("Claim Amount (USD)", 0.0, 1000000.0, 3000.0)

    submit = st.form_submit_button("Predict")

if submit:
    record = {
        "driver_age": driver_age,
        "driver_gender": driver_gender,
        "accident_location": accident_location,
        "vehicle_type": vehicle_type,
        "model": vehicle_model,
        "year": year,
        "value": value,
        "claim_type": claim_type,
        "days_active": days_active,
        "previous_claims": previous_claims,
        "police_report": police_report,
        "amount": amount
    }

    try:
        X = preprocess_input(record)
        prob = model.predict_proba(X)[0][1]
        prob_display = round(prob * 100, 2)

        # Adjust probability if similar cases were mostly fraud
        sims = find_similar_cases(record, n=5)
        if not sims.empty and sims["is_fraud"].mean() > 0.5 and prob < 0.5:
            prob = min(prob + 0.30, 0.99)
            prob_display = round(prob * 100, 2)

        # Risk level
        if prob >= 0.7:
            risk = "🔴 **High Risk**"
        elif prob >= 0.4:
            risk = "🟠 **Medium Risk**"
        else:
            risk = "🟢 **Low Risk**"

        st.subheader("Result")
        st.metric("Fraud Probability", f"{prob_display}%")
        st.write(risk)

        if not sims.empty:
            st.markdown("### Similar Historical Cases")
            st.dataframe(sims)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

import streamlit as st
import pandas as pd
import joblib
import gdown
import os

st.set_page_config(page_title="Prédiction du diabète", layout="centered")
st.title("Prédiction du diabète")

# --- Google Drive files ---
MODEL_FILE_ID = "1qU-l1xE1OGQj6Z9CllZAEHSmNesCtW7E"
SCALER_FILE_ID = "1H6Rw9PkI0ohbFgrMoJa_Fr6pK84YB8cY"

MODEL_LOCAL = "decisiontree.pkl"
SCALER_LOCAL = "scaler.pkl"

# Download model if not exists
if not os.path.exists(MODEL_LOCAL):
    st.info("Téléchargement du modèle...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_LOCAL, quiet=False)

if not os.path.exists(SCALER_LOCAL):
    st.info("Téléchargement du scaler...")
    gdown.download(f"https://drive.google.com/uc?id={SCALER_FILE_ID}", SCALER_LOCAL, quiet=False)

# Load model and scaler
model = joblib.load(MODEL_LOCAL)
scaler = joblib.load(SCALER_LOCAL)

# --- Mapping dictionaries ---
gender_mapping = {"female": 0, "male": 1, "other": 2}
smoking_mapping = {
    "never": 0,
    "current": 1,
    "former": 2,
    "notcurrent": 3,
    "ever": 4,
    "passive": 5,
    "not_sure": 6,
}

# --- Form ---
with st.form("prediction_form"):
    gender = st.selectbox("Genre", ["male", "female", "other"])
    age = st.number_input("Âge", min_value=0, max_value=120, value=25)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    heart_disease = st.selectbox("Maladie cardiaque", [0, 1], format_func=lambda x: "Oui" if x == 1 else "Non")
    smoking_history = st.selectbox(
        "Historique tabac",
        ["never", "current", "former", "notcurrent", "ever", "passive", "not_sure"]
    )
    weight = st.number_input("Poids (kg)", min_value=0, max_value=200, value=73)
    height = st.number_input("Taille (cm)", min_value=50, max_value=250, value=182)
    bmi = weight / ((height / 100) ** 2)
    hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5)
    glycemia = st.number_input("Glycémie (g/L)", min_value=0, max_value=500, value=1.0)

    submitted = st.form_submit_button("Prédire")

# --- Prediction ---
if submitted:
    df = pd.DataFrame([{
        "gender": gender_mapping.get(gender, 2),
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_mapping.get(smoking_history, 0),
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glycemia,
    }])

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred_raw = model.predict(df_scaled)[0]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_scaled)[0][1]
    else:
        prob = float(pred_raw)

    prob = float(max(0, min(prob, 1)) * 100)
    prediction = "Diabétique" if pred_raw == 1 else "Non diabétique"

    st.success(f"Prédiction : {prediction}")
    st.info(f"Probabilité : {prob:.2f}%")

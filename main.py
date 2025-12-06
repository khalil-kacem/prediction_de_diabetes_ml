from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Load model + scaler
model = joblib.load("decisiontree.pkl")
scaler = joblib.load("scaler.pkl")

# FastAPI app
app = FastAPI()

# CORS (allows Next.js to request)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class PredictionInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_g: float


# Mappings
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

@app.post("/predict")
def predict(data: PredictionInput):
    df = pd.DataFrame([{
        "gender": gender_mapping.get(data.gender, 2),
        "age": data.age,
        "hypertension": data.hypertension,
        "heart_disease": data.heart_disease,
        "smoking_history": smoking_mapping.get(data.smoking_history, 0),
        "bmi": data.bmi,
        "HbA1c_level": data.HbA1c_level,
        "blood_glucose_level": data.blood_glucose_g,
    }])

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    pred_raw = model.predict(df_scaled)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_scaled)[0][1]
    else:
        prob = float(pred_raw)

    prob = float(max(0, min(prob, 1)) * 100)

    prediction = "Diabétique" if pred_raw == 1 else "Non diabétique"

    return {
        "prediction": prediction,
        "probability": prob
    }

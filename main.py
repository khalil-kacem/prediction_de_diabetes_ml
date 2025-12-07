import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# -----------------------------
#  PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Prédiction du diabète",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
#  SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("🩺 Diabetes Predictor")
    st.write("Remplissez les informations pour prédire si la personne est diabétique ou non.")
    st.markdown("---")
    st.info("🔍 Le modèle utilise un Decision Tree + scaling des valeurs.")

# -----------------------------
#  DOWNLOAD MODEL IF NEEDED
# -----------------------------
MODEL_FILE_ID = "1qU-l1xE1OGQj6Z9CllZAEHSmNesCtW7E"
SCALER_FILE_ID = "1H6Rw9PkI0ohbFgrMoJa_Fr6pK84YB8cY"

MODEL_LOCAL = "decisiontree.pkl"
SCALER_LOCAL = "scaler.pkl"

if not os.path.exists(MODEL_LOCAL):
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_LOCAL, quiet=False)

if not os.path.exists(SCALER_LOCAL):
    gdown.download(f"https://drive.google.com/uc?id={SCALER_FILE_ID}", SCALER_LOCAL, quiet=False)

model = joblib.load(MODEL_LOCAL)
scaler = joblib.load(SCALER_LOCAL)

# -----------------------------
#  MAPPINGS
# -----------------------------
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

# -----------------------------
#  MAIN TITLE
# -----------------------------
st.markdown(
    """
    <h1 style="text-align: center;">🧬 Prédiction du Diabète</h1>
    <p style="text-align: center; color: gray;">
        Analyse intelligente basée sur un modèle Machine Learning
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")

# -----------------------------
#  LAYOUT COLUMNS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Informations personnelles")
    gender = st.selectbox("Genre", ["male", "female", "other"])
    age = st.number_input("Âge", min_value=1, max_value=120, value=25)
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    heart_disease = st.selectbox("Maladie cardiaque", [0, 1], format_func=lambda x: "Oui" if x else "Non")

with col2:
    st.subheader("🩸 Santé & mesures")
    smoking_history = st.selectbox(
        "Historique tabac",
        ["never", "current", "former", "notcurrent", "ever", "passive", "not_sure"]
    )
    weight = st.number_input("Poids (kg)", 0, 250, 73)
    height = st.number_input("Taille (cm)", 50, 250, 182)
    
    bmi = round(weight / ((height / 100) ** 2), 2)
    st.markdown(f"**IMC calculé :** <span style='color:#4CAF50; font-size:18px;'>🟢 {bmi}</span>", unsafe_allow_html=True)

    hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
    glycemia = st.number_input("Glycémie (g/L)", 0.0, 5.0, 1.0)

# -----------------------------
#  PREDICTION BUTTON
# -----------------------------
st.markdown("---")
center = st.columns(3)[1]

with center:
    predict_btn = st.button("🔮 Prédire", use_container_width=True)

# -----------------------------
#  PREDICTION LOGIC
# -----------------------------
if predict_btn:
    input_df = pd.DataFrame([{
        "gender": gender_mapping[gender],
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_mapping[smoking_history],
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glycemia,
    }])

    scaled = scaler.transform(input_df)
    raw_pred = model.predict(scaled)[0]

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled)[0][1]
    else:
        prob = float(raw_pred)

    prob = max(0, min(prob, 1))

    # -------------------------
    #  RESULT DISPLAY
    # -------------------------
    st.markdown("## 🧾 Résultat de l'analyse")
    
    if raw_pred == 1:
        st.error("⚠️ **Diabétique**", icon="🚨")
    else:
        st.success("🟢 **Non diabétique**", icon="😊")

    st.write("### 🔢 Probabilité :")
    st.progress(prob)

    st.markdown(
        f"""
        <h3 style='text-align:center;'>
            🔍 Précision estimée : <b>{prob*100:.2f}%</b>
        </h3>
        """,
        unsafe_allow_html=True,
    )

    st.info("Analyse réalisée avec succès ✔️")

# -----------------------------
#  FOOTER
# -----------------------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray;'>
        Développé avec ❤️ en Streamlit | Machine Learning Decision Tree
    </p>
    """,
    unsafe_allow_html=True,
)

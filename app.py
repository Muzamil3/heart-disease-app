"""Heart Disease Prediction - Streamlit web app (uses svc_heart_trained_model.pkl from the notebook)."""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

MODEL_PATH = Path(__file__).parent / "svc_heart_trained_model.pkl"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def build_label_encoders():
    # Same fixed category lists as the notebook -> identical numeric codes
    return {
        "ChestPain": LabelEncoder().fit(["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"]),
        "Gender": LabelEncoder().fit(["Male", "Female"]),
        "ExerciseAngina": LabelEncoder().fit(["Yes", "No"]),
        "MajorVessels": LabelEncoder().fit(["Zero", "One", "Two", "Three"]),
    }


model = load_model()
encoders = build_label_encoders()

st.title("❤️ Heart Disease Prediction System")
st.markdown(
    "Enter the patient's details and the trained **Support Vector Classifier** will predict "
    "whether heart disease is likely.  \n"
    "*Trained on the UCI Heart Disease (Cleveland) data. Test accuracy: **85%**.*"
)
st.divider()

col1, col2 = st.columns(2)
with col1:
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    major_vessels = st.selectbox("Major Vessels Coloured by Fluoroscopy", ["Zero", "One", "Two", "Three"])

if st.button("Predict", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "ChestPain": [chest_pain], "Gender": [gender],
        "ExerciseAngina": [exercise_angina], "MajorVessels": [major_vessels],
    })
    encoded = pd.DataFrame({c: encoders[c].transform(user_input[c]) for c in user_input.columns})
    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ⚠️ HEART DISEASE LIKELY")
    else:
        st.success("### ✅ NO HEART DISEASE LIKELY")
    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("Educational demo built from a small dataset using only 4 inputs. Not a medical device - not for diagnosis.")

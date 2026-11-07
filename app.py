
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# 1. LOAD SAVED MACHINE LEARNING MODEL
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_diabetes_model.joblib"

model = joblib.load(MODEL_PATH)


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="wide"
)


# ==========================================
# 3. DASHBOARD HEADER
# ==========================================

st.title("🏥 Healthcare AI Assistant")
st.subheader("Diabetes Prediction System")

st.write(
    "Enter patient health details below to estimate "
    "the likelihood of diabetes using a Machine Learning model."
)

st.divider()


# ==========================================
# 4. PATIENT HEALTH INFORMATION
# ==========================================

st.header("Patient Health Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies", 0, 20, 1
    )

    glucose = st.number_input(
        "Glucose Level", 0, 300, 120
    )

    blood_pressure = st.number_input(
        "Blood Pressure", 0, 200, 70
    )

    skin_thickness = st.number_input(
        "Skin Thickness", 0, 100, 20
    )


with col2:
    insulin = st.number_input(
        "Insulin Level", 0, 900, 80
    )

    bmi = st.number_input(
        "BMI", 0.0, 100.0, 25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        0.0, 3.0, 0.5
    )

    age = st.number_input(
        "Age", 1, 120, 30
    )


st.divider()


# ==========================================
# 5. DIABETES PREDICTION
# ==========================================

if st.button("Predict Diabetes", type="primary"):

    # Prepare patient input
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    # Validate measurements that should not be zero
    invalid_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    zero_columns = [
        column
        for column in invalid_columns
        if input_data[column].iloc[0] == 0
    ]

    if zero_columns:

        st.warning(
            "Please enter valid non-zero values for: "
            + ", ".join(zero_columns)
        )

    else:

        # Generate prediction
        prediction = model.predict(input_data)[0]

        st.divider()
        st.header("Prediction Result")

        # Calculate prediction probability
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            diabetes_probability = probabilities[1]

            st.metric(
                label="Estimated Diabetes Probability",
                value=f"{diabetes_probability * 100:.2f}%"
            )

            st.caption(
                "The model uses a 50% classification threshold."
            )

        # Display predicted class
        if prediction == 1:

            st.error(
                "The model classified this input as "
                "diabetes-positive."
            )

        else:

            st.success(
                "The model classified this input as "
                "diabetes-negative."
            )

        # Medical disclaimer
        st.caption(
            "This is an experimental Machine Learning prediction, "
            "not a medical diagnosis. Please consult a qualified "
            "healthcare professional."
        )


# ==========================================
# 6. FOOTER
# ==========================================

st.divider()

st.caption(
    "Healthcare AI Assistant | Diabetes Prediction "
    "using Machine Learning"
)

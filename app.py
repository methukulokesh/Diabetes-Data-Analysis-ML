import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration must be the first Streamlit command.
st.set_page_config(
    page_title="Healthcare AI | Diabetes Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "best_diabetes_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 15% 0%,#17364b 0%,#0b1929 42%,#071321 100%);color:#eef7fa}
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:3rem}
[data-testid="stHeader"]{background:transparent}
.hero{padding:1.7rem 2rem;border:1px solid #294b60;border-radius:22px;background:linear-gradient(120deg,#153b50,#142b43);margin-bottom:1.2rem}
.eyebrow{color:#59ddc7;font-size:.78rem;font-weight:800;letter-spacing:.14em}
.hero h1{color:#f5fbff;font-size:2.35rem;margin:.35rem 0}
.hero p{color:#b9ceda;margin:0}
.panel{background:#122b40;border:1px solid #29495d;border-radius:16px;padding:1rem 1.2rem;margin-bottom:1rem}
.section{font-size:1.25rem;font-weight:750;color:#eff8fa;border-bottom:1px solid #2a4a5e;padding:.5rem 0 .8rem;margin:1.1rem 0}
div[data-testid="stNumberInput"] label{color:#dcebf2!important;font-weight:650}
div[data-testid="stNumberInput"] input{background:#10283b;color:#f5fbff}
.stButton>button{width:100%;border:0;border-radius:12px;padding:.8rem;background:linear-gradient(90deg,#53dbc4,#8be8ce);color:#06242b;font-weight:800}
.stButton>button:hover{border:1px solid #b0f6e9;color:#06242b}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<div class="eyebrow">HEALTHCARE ANALYTICS • MACHINE LEARNING</div>
<h1>Healthcare AI Assistant</h1>
<p>Diabetes prediction prototype using eight patient health measurements.</p>
</div>
""", unsafe_allow_html=True)

a,b,c=st.columns(3)
for col,label,value in [(a,"MODEL","Decision Tree"),(b,"INPUTS","8 health features"),(c,"PURPOSE","Educational prototype")]:
    with col:
        st.markdown(f'<div class="panel"><small style="color:#9fb9c9">{label}</small><h3 style="margin:.3rem 0;color:#f3fbff">{value}</h3></div>',unsafe_allow_html=True)

st.markdown('<div class="section">Patient information</div>',unsafe_allow_html=True)
left,right=st.columns(2,gap="large")
with left:
    st.markdown("**Metabolic measurements**")
    pregnancies=st.number_input("Pregnancies",0,20,1)
    glucose=st.number_input("Glucose level",0,300,120)
    blood_pressure=st.number_input("Blood pressure",0,200,70)
    skin_thickness=st.number_input("Skin thickness",0,100,20)
with right:
    st.markdown("**Additional measurements**")
    insulin=st.number_input("Insulin level",0,900,80)
    bmi=st.number_input("BMI",0.0,100.0,25.0,step=.1)
    dpf=st.number_input("Diabetes pedigree function",0.0,3.0,.5,step=.01)
    age=st.number_input("Age",1,120,30)

st.markdown('<div class="section">Model assessment</div>',unsafe_allow_html=True)
st.caption("Enter measurements from a reliable source. Selected fields are validated before prediction.")

if st.button("Run diabetes prediction",type="primary"):
    patient=pd.DataFrame([{
        "Pregnancies":pregnancies,"Glucose":glucose,"BloodPressure":blood_pressure,
        "SkinThickness":skin_thickness,"Insulin":insulin,"BMI":bmi,
        "DiabetesPedigreeFunction":dpf,"Age":age
    }])
    required_nonzero=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    invalid=[name for name in required_nonzero if patient.at[0,name]==0]
    if invalid:
        st.warning("Please provide non-zero values for: "+", ".join(invalid))
    else:
        prediction=int(model.predict(patient)[0])
        probabilities=model.predict_proba(patient)[0] if hasattr(model,"predict_proba") else None
        st.markdown('<div class="section">Prediction result</div>',unsafe_allow_html=True)
        result_col,prob_col=st.columns(2,gap="large")
        with result_col:
            if prediction==1:
                st.error("Model classification: positive class")
            else:
                st.success("Model classification: negative class")
        with prob_col:
            if probabilities is not None:
                positive_probability=float(probabilities[1])
                st.metric("Estimated model probability for positive class",f"{positive_probability*100:.2f}%")
                st.progress(min(max(positive_probability,0.0),1.0))
                st.caption("This is a model output, not a clinical risk estimate.")
        st.info("Educational prediction only. This is not a medical diagnosis.")

st.markdown("""
<div style="border-top:1px solid #29495d;padding-top:1rem;color:#a9bdca;font-size:.82rem">
<strong>Medical disclaimer:</strong> This prototype uses a limited historical dataset and has not been clinically validated.
Do not use it to diagnose, treat or make healthcare decisions.
</div>
""",unsafe_allow_html=True)

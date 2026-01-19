import pickle
import streamlit as st
import numpy as np

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Loan Approval Credit Risk Predictor",
    layout="centered"
)

st.title("Loan Approval Credit Risk Predictor")

# ----------------------------
# Load Model & Scaler
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("Models/XgBoost.pkl", "rb"))
    scaler = pickle.load(open("Models/Scaler_new.pkl", "rb"))
    return model, scaler

model, standard_scaler = load_artifacts()

# ----------------------------
# Input Form
# ----------------------------
with st.form("loan_form"):
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    education = st.selectbox("Education", options=[0, 1], help="0: Not Graduate, 1: Graduate")
    self_employed = st.selectbox("Self Employed", options=[0, 1], help="0: No, 1: Yes")
    income_annum = st.number_input("Annual Income", min_value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.number_input("Loan Term (months)", min_value=0)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)

    submitted = st.form_submit_button("Predict Loan Approval")

# ----------------------------
# Prediction Logic
# ----------------------------
if submitted:
    try:
        input_data = np.array([[
            no_of_dependents,
            education,
            self_employed,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value
        ]])

        scaled_data = standard_scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        if int(prediction) == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

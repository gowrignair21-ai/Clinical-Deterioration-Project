import streamlit as st
import pandas as pd
import joblib 
from scipy.stats.mstats import winsorize 

# Load the pre-trained model and scalers
rf_model = joblib.load('rf_model.pkl')
standard_scaler = joblib.load('standard_scaler.pkl')
robust_scaler = joblib.load('robust_scaler.pkl')

# Define the columns that were scaled by StandardScaler and RobustScaler
ss_cols = ['heart_rate', 'respiratory_rate', 'temperature_c', 'wbc_count', 
           'creatinine', 'hemoglobin']
rbt_cols = ['spo2_pct', 'lactate', 'crp_level','oxygen_flow']

# Define winsorization limits used during training
winsor_limits = {
    'heart_rate': [0.01, 0.05],
    'respiratory_rate': [0, 0.05],
    'spo2_pct': [0.1, 0],
    'temperature_c': [0.01, 0.05],
    'systolic_bp': [0.05, 0.05],
    'diastolic_bp': [0.05, 0.05],
    'oxygen_flow': [0.05, 0.05],
    'wbc_count': [0.01, 0.07],
    'lactate': [0, 0.15],
    'creatinine': [0, 0.07],
    'crp_level': [0, 0.1],
    'hemoglobin': [0.05, 0.01]
}

expected_features = ['hour_from_admission', 'heart_rate', 'respiratory_rate', 'spo2_pct',
                     'temperature_c', 'systolic_bp', 'diastolic_bp', 'oxygen_flow',
                     'mobility_score', 'nurse_alert', 'wbc_count', 'lactate',
                     'creatinine', 'crp_level', 'hemoglobin', 'sepsis_risk_score', 'age',
                     'comorbidity_index', 'oxygen_device_hfnc', 'oxygen_device_mask',
                     'oxygen_device_nasal', 'oxygen_device_niv', 'oxygen_device_none',
                     'gender_F', 'gender_M', 'admission_type_ED',
                     'admission_type_Elective', 'admission_type_Transfer']

st.set_page_config(page_title="Hospital Deterioration Prediction", page_icon="🏥", layout="wide")

st.title("🏥 Hospital Deterioration Prediction")
st.markdown("Enter patient details to predict the likelihood of deterioration within the next 12 hours.")

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📋 Patient Background")
    
    oxygen_device_options = ['none', 'nasal', 'mask', 'hfnc', 'niv']
    oxygen_device = st.selectbox("Oxygen Device", oxygen_device_options)
    
    gender_options = ['M', 'F']
    gender = st.selectbox("Gender", gender_options)
    
    admission_type_options = ['Elective', 'Transfer', 'ED']
    admission_type = st.selectbox("Admission Type", admission_type_options)
    
    st.markdown("### 📊 Clinical Measurements")
    
    col1a, col1b = st.columns(2)
    with col1a:
        heart_rate = st.number_input("Heart Rate", min_value=40.0, max_value=180.0, value=125.0, step=0.1)
        respiratory_rate = st.number_input("Respiratory Rate", min_value=8.0, max_value=45.0, value=28.0, step=0.1)
        spo2_pct = st.number_input("SpO2 Percentage", min_value=70.0, max_value=100.0, value=88.0, step=0.1)
        temperature_c = st.number_input("Temperature (C)", min_value=35.0, max_value=41.0, value=38.5, step=0.1)
        systolic_bp = st.number_input("Systolic BP", min_value=70.0, max_value=185.0, value=85.0, step=0.1)
        diastolic_bp = st.number_input("Diastolic BP", min_value=40.0, max_value=110.0, value=50.0, step=0.1)
    with col1b:
        oxygen_flow = st.number_input("Oxygen Flow", min_value=0.0, max_value=60.0, value=5.0, step=0.1)
        wbc_count = st.number_input("WBC Count", min_value=2.0, max_value=30.0, value=16.0, step=0.1)
        lactate = st.number_input("Lactate", min_value=0.5, max_value=8.0, value=3.5, step=0.01)
        creatinine = st.number_input("Creatinine", min_value=0.4, max_value=4.5, value=2.1, step=0.01)
        crp_level = st.number_input("CRP Level", min_value=0.0, max_value=250.0, value=150.0, step=0.1)
        hemoglobin = st.number_input("Hemoglobin", min_value=7.0, max_value=17.0, value=10.5, step=0.1)

with col2:
    st.markdown("### 👤 Patient Information")
    
    col2a, col2b = st.columns(2)
    with col2a:
        hour_from_admission = st.number_input("Hours From Admission", min_value=0, max_value=71, value=8)
        age = st.number_input("Age", min_value=18, max_value=90, value=72)
    with col2b:
        mobility_score = st.slider("Mobility Score (0-4)", 0, 4, 1)
        comorbidity_index = st.slider("Comorbidity Index (0-8)", 0, 8, 3)
    
    nurse_alert = st.radio("Nurse Alert Triggered", [0, 1], index=1, horizontal=True)
    sepsis_risk_score = st.number_input("Sepsis Risk Score", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

# Prediction button centered
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Deterioration", use_container_width=True, type="primary")

# Result display area
if predict_button:
    try:
        # Collect inputs into a DataFrame
        input_data = {
            'hour_from_admission': [hour_from_admission],
            'heart_rate': [heart_rate],
            'respiratory_rate': [respiratory_rate],
            'spo2_pct': [spo2_pct],
            'temperature_c': [temperature_c],
            'systolic_bp': [systolic_bp],
            'diastolic_bp': [diastolic_bp],
            'oxygen_flow': [oxygen_flow],
            'mobility_score': [mobility_score],
            'nurse_alert': [nurse_alert],
            'wbc_count': [wbc_count],
            'lactate': [lactate],
            'creatinine': [creatinine],
            'crp_level': [crp_level],
            'hemoglobin': [hemoglobin],
            'sepsis_risk_score': [sepsis_risk_score],
            'age': [age],
            'comorbidity_index': [comorbidity_index],
            'oxygen_device': [oxygen_device],
            'gender': [gender],
            'admission_type': [admission_type]
        }
        input_df = pd.DataFrame(input_data)

        # Apply winsorization
        for col, limits in winsor_limits.items():
            if col in input_df.columns:
                input_df[col] = winsorize(input_df[col], limits=limits)

        # One-hot encoding
        input_df = pd.get_dummies(input_df, columns=['oxygen_device', 'gender', 'admission_type'])
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

        # Convert boolean columns to int
        bool_cols = input_df.select_dtypes(include='bool').columns
        if len(bool_cols) > 0:
            input_df[bool_cols] = input_df[bool_cols].astype(int)

        # Apply scaling
        input_df[ss_cols] = standard_scaler.transform(input_df[ss_cols])
        input_df[rbt_cols] = robust_scaler.transform(input_df[rbt_cols])     
                   
        prediction = rf_model.predict(input_df)
        prediction_proba = rf_model.predict_proba(input_df)

        st.subheader('Prediction Result:')
        st.write(f"**Prediction Value:** {prediction[0]}")
        st.write(f"**Probability of No Deterioration (0):** {prediction_proba[0][0]:.4f}")
        st.write(f"**Probability of Deterioration (1):** {prediction_proba[0][1]:.4f}")

        if prediction[0] == 1:
            st.error("⚠️ This patient is predicted to **deteriorate** within the next 12 hours.")
        else:
            st.success("✅ This patient is predicted to **not deteriorate** within the next 12 hours.")

        st.caption('Disclaimer: This prediction is for informational purposes only and should not replace professional medical advice.')
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.info("Please check your input values and try again.")

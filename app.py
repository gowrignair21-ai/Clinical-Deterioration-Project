import streamlit as st
import pandas as pd
import joblib
from scipy.stats.mstats import winsorize

rf_model = joblib.load('rf_model.pkl')
standard_scaler = joblib.load('standard_scaler.pkl')
robust_scaler = joblib.load('robust_scaler.pkl')

ss_cols = ['heart_rate', 'respiratory_rate', 'temperature_c', 'wbc_count',
           'creatinine', 'hemoglobin']
rbt_cols = ['spo2_pct', 'lactate', 'crp_level','oxygen_flow']

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

def calculate_derived_features(systolic_bp, diastolic_bp):
    """
    Calculate Mean Arterial Pressure (MAP) and Pulse Pressure (PP)
    
    MAP = Diastolic BP + (Systolic BP - Diastolic BP) / 3
    Pulse Pressure = Systolic BP - Diastolic BP
    """
    map_value = diastolic_bp + (systolic_bp - diastolic_bp) / 3
    pulse_pressure_value = systolic_bp - diastolic_bp
    return map_value, pulse_pressure_value

st.title('Hospital Deterioration Prediction')
st.write('Enter patient details to predict the likelihood of deterioration within the next 12 hours.')

st.sidebar.header('Patient Clinical Data')
hour_from_admission = st.sidebar.number_input('Hours From Admission', min_value=0, max_value=71, value=8)
heart_rate = st.sidebar.number_input('Heart Rate', min_value=40.0, max_value=180.0, value= 125.0, step=0.1)
respiratory_rate = st.sidebar.number_input('Respiratory Rate', min_value=8.0, max_value=45.0, value=28.0, step=0.1)
spo2_pct = st.sidebar.number_input('SpO2 Percentage', min_value=70.0, max_value=100.0, value=88.0, step=0.1)
temperature_c = st.sidebar.number_input('Temperature (C)', min_value=35.0, max_value=41.0, value=38.5, step=0.1)
systolic_bp = st.sidebar.number_input('Systolic BP', min_value=70.0, max_value=185.0, value=85.0, step=0.1)
diastolic_bp = st.sidebar.number_input('Diastolic BP', min_value=40.0, max_value=110.0, value=50.0, step=0.1)
oxygen_flow = st.sidebar.number_input('Oxygen Flow', min_value=0.0, max_value=60.0, value=5.0, step=0.1)
mobility_score = st.sidebar.slider('Mobility Score (0-4)', 0, 4, 1)
nurse_alert = st.sidebar.radio('Nurse Alert Triggered', [0, 1], index=1)
wbc_count = st.sidebar.number_input('WBC Count', min_value=2.0, max_value=30.0, value=16.0, step=0.1)
lactate = st.sidebar.number_input('Lactate', min_value=0.5, max_value=8.0, value=3.5, step=0.01)
creatinine = st.sidebar.number_input('Creatinine', min_value=0.4, max_value=4.5, value=2.1, step=0.01)
crp_level = st.sidebar.number_input('CRP Level', min_value=0.0, max_value=250.0, value=150.0, step=0.1)
hemoglobin = st.sidebar.number_input('Hemoglobin', min_value=7.0, max_value=17.0, value=10.5, step=0.1)
sepsis_risk_score = st.sidebar.number_input('Sepsis Risk Score', min_value=0.0, max_value=1.0, value=0.85, step=0.01)
age = st.sidebar.number_input('Age', min_value=18, max_value=90, value=72)
comorbidity_index = st.sidebar.slider('Comorbidity Index (0-8)', 0, 8, 3)
systolic_bp = st.sidebar.number_input('Systolic BP (mmHg)', min_value=70.0, max_value=185.0, value=85.0, step=0.1)
diastolic_bp = st.sidebar.number_input('Diastolic BP (mmHg)', min_value=40.0, max_value=110.0, value=50.0, step=0.1)

map_value, pp_value = calculate_derived_features(systolic_bp, diastolic_bp)
st.info(f"**Calculated Values:** Mean Arterial Pressure = {map_value:.1f} mmHg | Pulse Pressure = {pp_value:.1f} mmHg")

st.sidebar.header('Patient Background')
oxygen_device_options = ['none', 'nasal', 'mask', 'hfnc', 'niv']
oxygen_device = st.sidebar.selectbox('Oxygen Device', oxygen_device_options)
gender_options = ['M', 'F']
gender = st.sidebar.selectbox('Gender', gender_options)
admission_type_options = ['Elective', 'Transfer', 'ED']
admission_type = st.sidebar.selectbox('Admission Type', admission_type_options)

if st.button('Predict Deterioration'):
    try:
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

        map_values, pp_values = calculate_derived_features(systolic_bp, diastolic_bp)
        input_df['mean_arterial_pressure'] = map_values
        input_df['pulse_pressure'] = pp_values

        for col, limits in winsor_limits.items():
            if col in input_df.columns:
                input_df[col] = winsorize(input_df[col], limits=limits)

        input_df = pd.get_dummies(input_df, columns=['oxygen_device', 'gender', 'admission_type'])
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

        bool_cols = input_df.select_dtypes(include='bool').columns
        if len(bool_cols) > 0:
            input_df[bool_cols] = input_df[bool_cols].astype(int)

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

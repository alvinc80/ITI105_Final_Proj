import streamlit as st
import pickle
import numpy as np

# Load the model
with open('alzheimers_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
st.title('Forecasting Alzheimer\'s Disease')

# Input fields for features
gender = st.selectbox('Gender', ['Male', 'Female'])
ethnicity = st.slider('Ethnicity', 0, 100, 50)
educationlevel = st.slider('Education Level', 0, 3, 1)
bmi = st.slider('BMI', 15, 40, 22)
smoking = st.selectbox('Smoking', ['No', 'Yes'])
alcoholconsumpt = st.slider('Alcohol Consumption:', 0, 20, 10)
physicalact = st.slider('Physical Activity', 0, 10, 5)
dietqlt = st.slider('Diet Quality', 0, 10, 5)
sleepqlt = st.slider('Sleep Quality', 4, 10, 8)
systolicBP = st.slider('Systolic BP', 90, 180, 100)
diastolicBP = st.slider('Diastolic BP', 60, 120, 75)
cholesteroltotal = st.slider('Cholesterol Total', 150, 300, 190)
cholesterolLDL = st.slider('Cholesterol LDL', 50, 200, 90)
cholesterolHDL = st.slider('Cholesterol HDL', 20, 100, 60)
cholesterolTG = st.slider('Cholesterol Triglycerides', 50, 400, 140)
mmse = st.slider('Mini-Mental State Examination (MMSE)', 0, 30, 25)
functionalassmt = st.slider('Functional Assessment', 0, 10, 8)
memorycomplaints = st.selectbox('Memory Complaints', ['No', 'Yes'])
behavioralproblems = st.selectbox('Behavioral Problems', ['No', 'Yes'])
adl = st.slider('Activities of Daily Living (ADL)', 0, 10, 8)
agegroup = st.selectbox('Age Group', ['60-64', '65-69', '70-74', '75-79', '80-84', '85-90'])
total_symptom_score = st.slider('Total Symptom Score', 0, 4, 2)
total_risk_factors = st.slider('Total Risk Factors', 0, 5, 3)

# Convert categorical features to numerical if necessary
gender_numeric = 1 if gender == 'Female' else 0
smoking_numeric = 1 if smoking == 'Yes' else 0
memorycompl_numeric = 1 if memorycomplaints == 'Yes' else 0
behavioralprb_numeric = 1 if behavioralproblems == 'Yes' else 0

match gender:
    case '65-69':
        age_grp_numeric = 1
    case '70-74':
        age_grp_numeric = 2
    case '75-79':
        age_grp_numeric = 3
    case '80-84':
        age_grp_numeric = 4
    case '85-90':
        age_grp_numeric = 5
    case _:
        age_grp_numeric = 0
        
# Collect features
features = np.array([[gender_numeric, ethnicity, educationlevel, bmi, smoking_numeric, alcoholconsumpt, physicalact, dietqlt, sleepqlt, systolicBP, diastolicBP, cholesteroltotal, cholesterolLDL, cholesterolHDL, cholesterolTG, mmse, functionalassmt, memorycompl_numeric, behavioralprb_numeric, adl, age_grp_numeric, total_symptom_score, total_risk_factors]])

# Predict button
if st.button('Predict'):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    st.write(f'Prediction: {"Alzheimer\'s" if prediction[0] == 1 else "No Alzheimer\'s"}')
    st.write(f'Prediction Probability: {probability[0]}')
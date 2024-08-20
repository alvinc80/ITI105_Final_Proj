#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
import pickle
import numpy as np

# Load the model
with open('selected_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the Streamlit app
st.title('Alzheimer\'s Disease Prediction')

# Input fields for features
age = st.slider('Age', 0, 100, 50)
gender = st.selectbox('Gender', ['Male', 'Female'])
education = st.slider('Years of Education', 0, 20, 10)
test_score = st.slider('Cognitive Test Score', 0, 100, 50)

# Convert categorical features to numerical if necessary
gender_numeric = 1 if gender == 'Male' else 0

# Collect features
features = np.array([[age, gender_numeric, education, test_score]])

# Predict button
if st.button('Predict'):
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    st.write(f'Prediction: {"Alzheimer\'s" if prediction[0] == 1 else "No Alzheimer\'s"}')
    st.write(f'Prediction Probability: {probability[0]}')


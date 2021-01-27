"""
    For questions contact me on LinkedIn at 
    File name         : heart-disease-app.py
    Author            : Chong Shen Ng
    Date created      : 26/01/2021
    Date last modified: 27/01/2021
    Python Version    : 3.8
    Description       : Prototype app to predict heart disease in a person.
                        For queries, you can reach me on LinkedIn at
                        http://www.linkedin.com/in/chongshenng
"""
# Import necessary libraries
import streamlit as st
import numpy as np
import pickle
import time

st.title("Heart Disease Predictor")

#Note: You need 'Sex','FastingBloodSugar','RestingECG','ExercisedInduced'
#      This model is a prototype and can be further improved by adding
#      more features (age, resting blood pressure, etc ...), feature engineering,
#      and tuning the models using a grid-search.

def make_one(feature,feature_for_one):
    # Function to convert yes-no selection to ones and zeros
    return 1 if feature == feature_for_one else 0

st.markdown("Select from the following to get a heart disease prediction:")
sex = st.radio(
    "Is the patient male or female?", 
    ("Male", "Female"))
sex_binarised = make_one(sex,"Male")

fasting_blood_sugar = st.radio(
    "Is the patient diabetic? (Fasting blood sugar > 120mg/dl)?", 
    ("Yes","No"))
fasting_blood_sugar_binarised = make_one(fasting_blood_sugar,"Yes")

resting_ecg = st.radio(
    "Are the patient's' resting ECG results normal?", 
    ("Yes","No"))
resting_ecg_binarised = make_one(resting_ecg,"No")

exercised_induced_angina = st.radio(
    "During exercise, does the patient experience chest pains?", 
    ("Yes","No"))
exercised_induced_angina_binarised = make_one(exercised_induced_angina,"Yes")

# Now we construct the patient input into a readable format for the sklearn model object
patient_input = np.array([sex_binarised, 
                 fasting_blood_sugar_binarised, 
                 resting_ecg_binarised,
                 exercised_induced_angina_binarised]).reshape(1,-1)

# Give it a short pause before returning a result
with st.spinner():
    time.sleep(0.5)

# Now load the trained Naive-Bayes model 
filename = "./models/heartdisease_NB_model.p"
loaded_model = pickle.load(open(filename, 'rb'))
heart_disease_prediction_prob = loaded_model.predict_proba(patient_input)
heart_disease_prediction = loaded_model.predict(patient_input)  #1 means no heart disease
                                                                #0 means heart disease is likely

if heart_disease_prediction[0] == 1:
    st.markdown("Congratulations! You have no heart disease!")
else:
    st.write("I'm sorry, but it's likely you have a heart disease.")
import streamlit as st
import pickle
import os

print(os.getcwd())

model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(
    page_title="AI SocialMedia Prediction"
)

st.title("Student Prediction")

st.write(
    "Predict Daily Social Media Usage"
)

age = st.number_input(
    "Enter Age",
    18,
    70
)

Daily_Social_Media_Hours = st.number_input(
    "Enter Social Media Hours",
    0.0,
    24.0
)

Daily_AI_Tool_Usage_Hours = st.number_input(
    "Enter AI Usage Hours",
    0.0,
    24.0
)

Sleep_Hours = st.number_input(
    "Enter Sleeping Hours",
    0.0,
    24.0
)

Physical_Activity_Hours = st.number_input(
    "Enter Activity Hours",
    0.0,
    24.0
)

Mental_Health_Score = st.number_input(
    "Enter Mental Health Score",
    0,
    100
)

Physical_Health_Score = st.number_input(
    "Enter Physical Health Score",
    0,
    100
)

if st.button("Predict"):

    result = model.predict([
        [
            age,
            Daily_Social_Media_Hours,
            Daily_AI_Tool_Usage_Hours,
            Sleep_Hours,
            Physical_Activity_Hours,
            Mental_Health_Score,
            Physical_Health_Score
        ]
    ])

    if result[0] == 1:
        st.success(" High Social Media Usage ✅")
    else:
        st.error(" Low Social Media Usage ❌")
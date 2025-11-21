import streamlit as st
import joblib
import numpy as np
import os
from openai import OpenAI

# ---------------- PAGE SETUP ---------------- #
st.set_page_config(page_title="Diabetes Prediction + AI Assistant", layout="centered")
st.title("🩺 AI Diabetes Prediction App + Health Assistant")
st.write("This app predicts diabetes risk and provides diet plans, doctor recommendations, and general health guidance.")

# ---------------- LOAD MODEL & SCALER ---------------- #
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# ---------------- OPENAI CLIENT (from environment) ---------------- #
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ OPENAI_API_KEY not found in environment variables!")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------- INPUT FORM ---------------- #
st.header("Enter Patient Details:")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=85)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

# ---------------- PREDICTION ---------------- #
prediction_result = None
probability = None

if st.button("🔍 Predict Diabetes Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔔 Prediction Result:")
    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Diabetes! (Probability: {probability*100:.2f}%)")
        prediction_result = "High Risk"
    else:
        st.success(f"✅ Low Risk of Diabetes (Probability: {probability*100:.2f}%)")
        prediction_result = "Low Risk"

    st.caption("Disclaimer: Machine-learning model only. Not a medical diagnosis.")

# ---------------- CHATBOT ---------------- #
st.markdown("---")
st.header("🤖 AI Health Assistant (Diet, Exercise, Doctor Advice)")

# Chat instructions (system prompt)
system_prompt = """
You are a diabetes health assistant. Your tasks:
1. Provide diabetes awareness and basic guidance.
2. If user asks, generate:
   - Diet plan (veg or non-veg)
   - Exercise routine
   - Lifestyle changes
3. Based on symptoms or risk, recommend the correct doctor type:
   - Endocrinologist (diabetes specialist)
   - Dietitian
   - Cardiologist (heart issues)
   - Nephrologist (kidney issues)
   - Ophthalmologist (eye problems)
4. Be friendly and simple. No medical diagnosis. Education only.
5. If user gives prediction probability, explain it clearly.
"""

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": system_prompt}
    ]

# Show chat history
for msg in st.session_state.messages[1:]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_msg = st.chat_input("Ask me for diet plan, exercise, doctor recommendation or diabetes doubts...")

if user_msg:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_msg})

    # Generate AI response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages
    )

    bot_msg = response.choices[0].message["content"]

    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": bot_msg})

    # Display response
    with st.chat_message("assistant"):
        st.write(bot_msg)

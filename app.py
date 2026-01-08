# ============================================================
# ADVANCED DIABETES AI ASSISTANT
# Developed by: Kumar GK (Student Project)
# ============================================================

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from openai import OpenAI
from fpdf import FPDF

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Advanced Diabetes AI Assistant",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background:#0b1220; color:#e5e7eb; }
.card { background:#111827; padding:18px; border-radius:16px; margin-bottom:16px; }
.chat-user { background:#065f46; padding:10px; border-radius:12px; margin:6px 0; }
.chat-bot { background:#1f2937; padding:10px; border-radius:12px; margin:6px 0; }
h1,h2,h3 { color:#7dd3fc; }
.footer { text-align:center; color:#9ca3af; font-size:14px; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="card">
<h1>🩺 Advanced Diabetes AI Assistant</h1>
<p>
Built by <b>Kumar GK</b> | Student Project <br>
An educational AI system for diabetes awareness & learning
</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- OPENAI ----------------
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in Streamlit Secrets")
    st.stop()

client = OpenAI()

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
You are an AI healthcare assistant for diabetes education.
This is a STUDENT PROJECT for learning purposes.

Rules:
- No medicine dosage
- No diagnosis confirmation
- Always suggest consulting doctors
- Adjust advice for kids, adults, elderly
"""

# ---------------- DATABASE ----------------
HOSPITAL_DB = {
    "Bangalore": {
        "government": {
            "diabetes": ["Victoria Hospital – KR Market"],
            "heart": ["Jayadeva Hospital – Bannerghatta Road"],
            "kidney": ["Victoria Hospital – Nephrology"],
            "eye": ["Minto Eye Hospital – Chamrajpet"]
        },
        "private": {
            "diabetes": ["Apollo Hospital – Bannerghatta Road"],
            "heart": ["Narayana Health – Bommasandra"],
            "kidney": ["Manipal Hospital – Old Airport Road"],
            "eye": ["Narayana Nethralaya – Rajajinagar"]
        }
    },
    "Delhi": {
        "government": {
            "diabetes": ["AIIMS – Ansari Nagar"],
            "heart": ["AIIMS Cardiology"],
            "kidney": ["Safdarjung Hospital"],
            "eye": ["Dr RP Centre – AIIMS"]
        },
        "private": {
            "diabetes": ["Max Hospital – Saket"],
            "heart": ["Fortis Escorts – Okhla"],
            "kidney": ["BLK Hospital"],
            "eye": ["Centre for Sight – Dwarka"]
        }
    }
}

MEDICINES = [
    "Metformin",
    "Glimepiride",
    "Sitagliptin",
    "Insulin (Injection)",
    "Atorvastatin",
    "Losartan",
    "Vitamin B12"
]

SYMPTOM_MAP = {
    "frequent urination": "Diabetes",
    "burning feet": "Diabetic Neuropathy",
    "blurred vision": "Diabetic Retinopathy",
    "chest pain": "Heart Disease",
    "leg swelling": "Kidney Disease"
}

# ---------------- HELPERS ----------------
def extract_city(text):
    for c in HOSPITAL_DB.keys():
        if c.lower() in text:
            return c
    return None

def detect_organ(text):
    if "heart" in text:
        return "heart"
    if "kidney" in text:
        return "kidney"
    if "eye" in text or "vision" in text:
        return "eye"
    return "diabetes"

def detect_type(text):
    if "government" in text or "govt" in text:
        return "government"
    if "private" in text:
        return "private"
    return None

def bmi_age_advice(bmi, age):
    if age < 18:
        return "\n🧒 Child case: Likely Type 1 diabetes. Pediatric endocrinologist required."
    if age >= 60:
        return "\n👴 Elderly care: Prefer multi-speciality hospital, heart & kidney monitoring."
    if bmi >= 30:
        return "\n⚠️ Obesity risk: Lifestyle clinic + cardiology support advised."
    return "\n✅ General hospital with endocrinology OPD is sufficient."

# ---------------- PREDICTION ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔍 Diabetes Risk Prediction")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 1, 120, 33)

if gender == "Female" and age >= 12:
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
else:
    pregnancies = 0

c1,c2,c3,c4 = st.columns(4)
glucose = c1.number_input("Glucose", 0, 300, 120)
bp = c2.number_input("Blood Pressure", 0, 200, 70)
skin = c3.number_input("Skin Thickness", 0, 100, 20)
insulin = c4.number_input("Insulin", 0, 900, 80)

c5,c6,c7 = st.columns(3)
bmi = c5.number_input("BMI", 0.0, 60.0, 25.0)
dpf = c6.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Predict Diabetes Risk"):
    X = scaler.transform([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.success(f"Risk: {'High' if pred else 'Low'} ({prob*100:.2f}%)")

    st.session_state.history.append({
        "Age": age,
        "Gender": gender,
        "BMI": bmi,
        "Glucose": glucose,
        "Risk": "High" if pred else "Low",
        "Probability (%)": round(prob*100, 2)
    })

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CHATBOT ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Health Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

for m in st.session_state.messages:
    if m["role"] == "user":
        st.markdown(f"<div class='chat-user'>👤 {m['content']}</div>", unsafe_allow_html=True)
    elif m["role"] == "assistant":
        st.markdown(f"<div class='chat-bot'>🤖 {m['content']}</div>", unsafe_allow_html=True)

query = st.text_input("Ask about hospitals, medicines, heart, kidney, eye, kids, gym...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    ai = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages
    )

    bot = ai.choices[0].message.content
    text = query.lower()

    if "medicine" in text:
        bot += "\n\n💊 Common Medicines:\n" + "\n".join([f"- {m}" for m in MEDICINES])

    for s, d in SYMPTOM_MAP.items():
        if s in text:
            bot += f"\n🧠 Possible condition: {d}"

    city = extract_city(text)
    organ = detect_organ(text)
    htype = detect_type(text)

    if city:
        bot += f"\n\n🏥 Hospitals in {city} ({organ.title()}):\n"
        if htype:
            hs = HOSPITAL_DB[city][htype].get(organ, [])
        else:
            hs = (
                HOSPITAL_DB[city]["government"].get(organ, []) +
                HOSPITAL_DB[city]["private"].get(organ, [])
            )
        for h in hs:
            bot += f"- {h}\n"

    bot += bmi_age_advice(bmi, age)
    bot += "\n\n⚠️ Project Disclaimer: This is a student project for educational purposes only."

    st.session_state.messages.append({"role": "assistant", "content": bot})

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HISTORY ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧾 Patient History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.table(df)

    if st.button("Download History PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0,10,"Diabetes Prediction History", ln=True)
        for _, row in df.iterrows():
            pdf.cell(0,8,str(dict(row)), ln=True)
        pdf.output("history.pdf")

        with open("history.pdf", "rb") as f:
            st.download_button("Download PDF", f, file_name="history.pdf")
else:
    st.info("No history yet")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
<b>Developed by Kumar GK</b> <br>
Student Mini Project | For Academic & Learning Purpose Only <br>
Not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)

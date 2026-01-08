# =========================================================
# ADVANCED DIABETES AI ASSISTANT – FULL FEATURE APP
# =========================================================

import streamlit as st
import numpy as np
import joblib
import os
import tempfile
import pandas as pd
from openai import OpenAI

# ========================= UI CONFIG ======================
st.set_page_config(page_title="Diabetes AI Assistant", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp { background:#0b1220; color:#e5e7eb; }
.card { background:#111827; padding:18px; border-radius:16px; margin-bottom:16px; }
.chat-user { background:#065f46; padding:10px; border-radius:12px; margin:6px 0; }
.chat-bot { background:#1f2937; padding:10px; border-radius:12px; margin:6px 0; }
h1,h2,h3 { color:#7dd3fc; }
</style>
""", unsafe_allow_html=True)

# ========================= LOAD MODEL =====================
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# ========================= OPENAI =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """
You are an Advanced Diabetes Healthcare AI Assistant for INDIA.
Reply in the SAME language as the user (English / Kannada / Hindi).

Rules:
- No medicine dosage
- No real-time claims
- Always medical disclaimer
- Be structured & supportive
"""

# ========================= DATABASE ======================
HOSPITAL_DB = {
    "Bangalore": {
        "government": {
            "diabetes": ["Victoria Hospital – KR Market"],
            "heart": ["Jayadeva Hospital – Bannerghatta Rd"],
            "kidney": ["Victoria Hospital – Nephrology"],
            "eye": ["Minto Eye Hospital – Chamrajpet"]
        },
        "private": {
            "diabetes": ["Apollo Hospital – Bannerghatta Rd"],
            "heart": ["Narayana Health – Bommasandra"],
            "kidney": ["Manipal Hospital – Old Airport Rd"],
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
    },
    "Mumbai": {
        "government": {
            "diabetes": ["JJ Hospital – Byculla"],
            "heart": ["KEM Hospital – Parel"],
            "kidney": ["Sion Hospital"],
            "eye": ["JJ Eye Dept"]
        },
        "private": {
            "diabetes": ["Lilavati Hospital – Bandra"],
            "heart": ["Kokilaben Hospital – Andheri"],
            "kidney": ["Hiranandani Hospital – Powai"],
            "eye": ["Aditya Jyot Eye Hospital"]
        }
    }
}

MEDICINES = [
    "Metformin", "Glimepiride", "Sitagliptin",
    "Insulin", "Atorvastatin", "Losartan", "Vitamin B12"
]

SYMPTOM_MAP = {
    "frequent urination": "Diabetes",
    "burning feet": "Diabetic Neuropathy",
    "blurred vision": "Diabetic Retinopathy",
    "chest pain": "Heart Disease",
    "leg swelling": "Kidney Disease"
}

# ========================= HELPERS ========================
def extract_city(text):
    for c in HOSPITAL_DB.keys():
        if c.lower() in text:
            return c
    return None

def detect_organ(text):
    if "heart" in text: return "heart"
    if "kidney" in text: return "kidney"
    if "eye" in text or "vision" in text: return "eye"
    return "diabetes"

def detect_type(text):
    if "government" in text or "govt" in text: return "government"
    if "private" in text: return "private"
    return None

def detect_language(text):
    if any(ch in text for ch in "ಕನ್ನಡ"): return "kannada"
    if any(ch in text for ch in "अस्पतालडॉक्टरइलाज"): return "hindi"
    return "english"

def bmi_age_advice(bmi, age):
    if bmi >= 30 or age >= 60:
        return "🏥 Prefer multi-speciality hospital with ICU."
    elif bmi >= 25:
        return "🏥 Hospital with diabetes + cardiology dept recommended."
    return "🏥 General hospital with endocrinology OPD sufficient."

# ========================= HEADER =========================
st.markdown("<div class='card'><h1>🩺 Diabetes AI Assistant</h1><p>Prediction • Chat • Hospitals • Voice • History</p></div>", unsafe_allow_html=True)

# ========================= PREDICTION =====================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🔍 Diabetes Risk Prediction")

c1,c2,c3,c4 = st.columns(4)
preg = c1.number_input("Pregnancies",0,20,2)
glu  = c2.number_input("Glucose",0,300,120)
bp   = c3.number_input("Blood Pressure",0,200,70)
skin = c4.number_input("Skin Thickness",0,100,20)

c5,c6,c7,c8 = st.columns(4)
ins  = c5.number_input("Insulin",0,900,80)
bmi  = c6.number_input("BMI",0.0,60.0,25.0)
dpf  = c7.number_input("DPF",0.0,3.0,0.5)
age  = c8.number_input("Age",1,120,33)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("Predict"):
    X = scaler.transform([[preg,glu,bp,skin,ins,bmi,dpf,age]])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    st.success(f"Risk: {'High' if pred else 'Low'} ({prob*100:.2f}%)")

    st.session_state.history.append({
        "Age": age, "BMI": bmi, "Glucose": glu,
        "Risk": "High" if pred else "Low",
        "Probability": round(prob*100,2)
    })
st.markdown("</div>", unsafe_allow_html=True)

# ========================= CHATBOT ========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Health Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"system","content":SYSTEM_PROMPT}]

for m in st.session_state.messages:
    if m["role"]=="user":
        st.markdown(f"<div class='chat-user'>👤 {m['content']}</div>", unsafe_allow_html=True)
    elif m["role"]=="assistant":
        st.markdown(f"<div class='chat-bot'>🤖 {m['content']}</div>", unsafe_allow_html=True)

query = st.text_input("Ask about hospitals, medicines, heart, kidney, eye…")

if query:
    st.session_state.messages.append({"role":"user","content":query})
    ai = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages)
    bot = ai.choices[0].message.content

    text = query.lower()
    city = extract_city(text)
    organ = detect_organ(text)
    htype = detect_type(text)

    if "medicine" in text:
        bot += "\n💊 Common Medicines:\n" + "\n".join([f"- {m}" for m in MEDICINES])

    for s,d in SYMPTOM_MAP.items():
        if s in text:
            bot += f"\n🧠 Possible condition: {d}"

    if city:
        bot += f"\n🏥 Hospitals in {city} ({organ.title()}):\n"
        if htype:
            hs = HOSPITAL_DB[city][htype].get(organ,[])
        else:
            hs = HOSPITAL_DB[city]["government"].get(organ,[]) + HOSPITAL_DB[city]["private"].get(organ,[])
        for h in hs:
            bot += f"- {h}\n"

        bot += bmi_age_advice(bmi, age)

    bot += "\n⚠️ Disclaimer: Educational only. Consult doctors."

    st.session_state.messages.append({"role":"assistant","content":bot})
    st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ========================= HISTORY ========================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧾 Patient History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.table(df)
    st.download_button("Download History CSV", df.to_csv(index=False), "history.csv")
else:
    st.info("No history yet")

st.markdown("</div>", unsafe_allow_html=True)

# ========================= FOOTER =========================
st.markdown("<p style='text-align:center;color:#9ca3af'>Educational only • Not medical advice</p>", unsafe_allow_html=True)

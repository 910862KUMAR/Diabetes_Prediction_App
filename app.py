# ============================================================
# ADVANCED DIABETES AI ASSISTANT
# Designed & Developed by Kumar GK
# Student Academic Project
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

# ---------------- LANGUAGE SELECT ----------------
language = st.selectbox(
    "🌐 Language / ಭಾಷೆ / भाषा",
    ["English", "Kannada", "Hindi"]
)

CAPTIONS = {
    "English": {
        "title": "Advanced Diabetes AI Assistant",
        "subtitle": "Designed & Developed by Kumar GK",
        "tagline": "Student Academic Project — Intelligent Healthcare Assistant",
        "footer": "This application is developed only for educational purposes and does not replace medical advice."
    },
    "Kannada": {
        "title": "ಅಡ್ವಾನ್ಸ್ ಡಯಾಬಿಟಿಸ್ ಎಐ ಸಹಾಯಕ",
        "subtitle": "ವಿನ್ಯಾಸಗೊಳಿಸಿ ಅಭಿವೃದ್ಧಿಪಡಿಸಿದವರು: ಕುಮಾರ್ ಜಿಕೆ",
        "tagline": "ವಿದ್ಯಾರ್ಥಿ ಶೈಕ್ಷಣಿಕ ಪ್ರಾಜೆಕ್ಟ್ — ಬುದ್ಧಿವಂತ ಆರೋಗ್ಯ ಸಹಾಯಕ",
        "footer": "ಈ ಅಪ್ಲಿಕೇಶನ್ ಶೈಕ್ಷಣಿಕ ಉದ್ದೇಶಗಳಿಗೆ ಮಾತ್ರ ಮತ್ತು ವೈದ್ಯಕೀಯ ಸಲಹೆಗೆ ಪರ್ಯಾಯವಲ್ಲ."
    },
    "Hindi": {
        "title": "एडवांस्ड डायबिटीज एआई असिस्टेंट",
        "subtitle": "विकसितकर्ता: कुमार जीके",
        "tagline": "छात्र शैक्षणिक परियोजना — स्मार्ट हेल्थकेयर सहायक",
        "footer": "यह एप्लिकेशन केवल शैक्षणिक उद्देश्य के लिए है और चिकित्सकीय सलाह का विकल्प नहीं है।"
    }
}

# ---------------- HEADER ----------------
st.markdown(f"""
<div class="card">
<h1>{CAPTIONS[language]['title']}</h1>
<p>
<b>{CAPTIONS[language]['subtitle']}</b><br>
{CAPTIONS[language]['tagline']}
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

# ---------------- SYSTEM PROMPT (LANGUAGE AWARE) ----------------
SYSTEM_PROMPT = f"""
You are a calm, professional healthcare AI assistant for diabetes education.
This is a STUDENT ACADEMIC PROJECT.

IMPORTANT:
- Reply ONLY in {language}
- Keep answers simple, structured, and calm (Alexa-style)
- No medicine dosage
- No final diagnosis
- Always suggest consulting doctors
- Adjust advice for kids (Type 1), adults (Type 2), and elderly

End every reply with a short safety note.
"""

# ---------------- DATABASE ----------------
HOSPITAL_DB = {
    "Bangalore": {
        "diabetes": ["Victoria Hospital – KR Market", "Apollo Hospital – Bannerghatta Road"],
        "heart": ["Jayadeva Hospital – Bannerghatta Road"],
        "kidney": ["Victoria Hospital – Nephrology"],
        "eye": ["Minto Eye Hospital – Chamrajpet"]
    },
    "Delhi": {
        "diabetes": ["AIIMS – Ansari Nagar", "Max Hospital – Saket"],
        "heart": ["AIIMS Cardiology"],
        "kidney": ["Safdarjung Hospital"],
        "eye": ["Dr RP Centre – AIIMS"]
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

# ---------------- HELPERS ----------------
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

def age_based_note(age):
    if age < 18:
        return {
            "English": "🧒 This appears to be a child case. Type 1 diabetes is common. Pediatric endocrinologist consultation is important.",
            "Kannada": "🧒 ಇದು ಮಕ್ಕಳ ಪ್ರಕರಣವಾಗಿರಬಹುದು. ಟೈಪ್ 1 ಡಯಾಬಿಟಿಸ್ ಸಾಮಾನ್ಯ. ಮಕ್ಕಳ ಎಂಡೋಕ್ರೈನಾಲಜಿಸ್ಟ್ ಸಲಹೆ ಅಗತ್ಯ.",
            "Hindi": "🧒 यह बच्चों का मामला हो सकता है। टाइप 1 डायबिटीज सामान्य है। बाल रोग विशेषज्ञ से परामर्श जरूरी है।"
        }[language]
    if age >= 60:
        return {
            "English": "👴 Elderly care: heart, kidney, and BP monitoring is important.",
            "Kannada": "👴 ಹಿರಿಯರ ಆರೈಕೆ: ಹೃದಯ, ಕಿಡ್ನಿ ಮತ್ತು ರಕ್ತದೊತ್ತಡ ಮೇಲ್ವಿಚಾರಣೆ ಅಗತ್ಯ.",
            "Hindi": "👴 बुजुर्गों में हृदय, किडनी और बीपी की निगरानी जरूरी है।"
        }[language]
    return ""

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

c5,c6 = st.columns(2)
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
    st.session_state.messages = [{"role":"system","content":SYSTEM_PROMPT}]

for m in st.session_state.messages:
    if m["role"]=="user":
        st.markdown(f"<div class='chat-user'>👤 {m['content']}</div>", unsafe_allow_html=True)
    elif m["role"]=="assistant":
        st.markdown(f"<div class='chat-bot'>🤖 {m['content']}</div>", unsafe_allow_html=True)

query = st.text_input("Ask about diabetes, hospitals, medicines, kids, elderly care...")

if query:
    st.session_state.messages.append({"role":"user","content":query})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=st.session_state.messages
    )

    bot = response.choices[0].message.content
    text = query.lower()

    if "medicine" in text:
        bot += "\n\n💊 " + ("Common medicines:" if language=="English" else
                             "ಸಾಮಾನ್ಯ ಔಷಧಿಗಳು:" if language=="Kannada"
                             else "सामान्य दवाइयाँ:")
        for m in MEDICINES:
            bot += f"\n- {m}"

    city = extract_city(text)
    if city:
        organ = detect_organ(text)
        bot += f"\n\n🏥 Hospitals in {city} ({organ.title()}):"
        for h in HOSPITAL_DB[city][organ]:
            bot += f"\n- {h}"

    bot += "\n\n" + age_based_note(age)
    bot += "\n\n⚠️ " + CAPTIONS[language]["footer"]

    st.session_state.messages.append({"role":"assistant","content":bot})

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HISTORY + PDF ----------------
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
            pdf.multi_cell(0,8,str(dict(row)))
        pdf.output("history.pdf")

        with open("history.pdf","rb") as f:
            st.download_button("Download PDF", f, file_name="history.pdf")
else:
    st.info("No history yet")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(f"""
<div class="footer">
<b>Developed by Kumar GK</b><br>
Student Academic Project<br>
{CAPTIONS[language]['footer']}
</div>
""", unsafe_allow_html=True)

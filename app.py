# ============================================================
# ADVANCED DIABETES AI ASSISTANT (LOCKED FINAL VERSION)
# Developed by Kumar GK
# Student Academic Project
# ============================================================

import streamlit as st
import numpy as np
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
from fpdf import FPDF
from gtts import gTTS
from openai import OpenAI

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
.answer { background:#1f2937; padding:14px; border-radius:12px; }
.footer { text-align:center; color:#9ca3af; font-size:14px; }
</style>
""", unsafe_allow_html=True)

# ---------------- LANGUAGE ----------------
language = st.selectbox("🌐 Language / ಭಾಷೆ / भाषा", ["English", "Kannada", "Hindi"])

LANG_CODE = {"English": "en", "Kannada": "kn", "Hindi": "hi"}

# ---------------- CAPTIONS ----------------
CAPTIONS = {
    "English": {
        "title": "Advanced Diabetes AI Assistant",
        "subtitle": "Developed by Kumar GK",
        "tagline": "Student Academic Project — Intelligent Healthcare Assistant",
        "disclaimer": "Educational purpose only. Does not replace medical advice."
    },
    "Kannada": {
        "title": "ಅಡ್ವಾನ್ಸ್ ಡಯಾಬಿಟಿಸ್ ಎಐ ಸಹಾಯಕ",
        "subtitle": "ಅಭಿವೃದ್ಧಿಪಡಿಸಿದವರು: ಕುಮಾರ್ ಜಿಕೆ",
        "tagline": "ವಿದ್ಯಾರ್ಥಿ ಶೈಕ್ಷಣಿಕ ಪ್ರಾಜೆಕ್ಟ್ — ಬುದ್ಧಿವಂತ ಆರೋಗ್ಯ ಸಹಾಯಕ",
        "disclaimer": "ಶೈಕ್ಷಣಿಕ ಉದ್ದೇಶಕ್ಕಾಗಿ ಮಾತ್ರ. ವೈದ್ಯಕೀಯ ಸಲಹೆಗೆ ಪರ್ಯಾಯವಲ್ಲ."
    },
    "Hindi": {
        "title": "एडवांस्ड डायबिटीज एआई असिस्टेंट",
        "subtitle": "विकसितकर्ता: कुमार जीके",
        "tagline": "छात्र शैक्षणिक परियोजना — स्मार्ट हेल्थकेयर सहायक",
        "disclaimer": "केवल शैक्षणिक उद्देश्य के लिए। चिकित्सकीय सलाह का विकल्प नहीं।"
    }
}

# ---------------- HEADER ----------------
st.markdown(f"""
<div class="card">
<h1>{CAPTIONS[language]['title']}</h1>
<p><b>{CAPTIONS[language]['subtitle']}</b><br>
{CAPTIONS[language]['tagline']}</p>
</div>
""", unsafe_allow_html=True)

# ---------------- VOICE ----------------
def speak(text):
    tts = gTTS(text=text, lang=LANG_CODE[language])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)

# ---------------- OPENAI CLIENT ----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ai_fallback(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a health assistant. "
                    "Answer clearly and safely. "
                    "Do not give medical diagnosis or dosage. "
                    "Always mention this is for educational purposes only."
                )
            },
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl"), joblib.load("scaler.pkl")

model, scaler = load_model()

# ---------------- DIABETES PREDICTION ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧪 Diabetes Risk Prediction")

age = st.number_input("Age", 1, 120, 35)

gender = st.selectbox("Gender", ["Male", "Female"])
pregnancies = st.number_input("Pregnancies", 0, 20, 0) if gender == "Female" and age >= 15 else 0

glucose = st.number_input("Glucose Level", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

if st.button("🔍 Predict Diabetes"):
    X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)[0][1] * 100

    risk = "High" if prob > 70 else "Medium" if prob > 40 else "Low"
    st.success(f"Risk Level: {risk} ({prob:.2f}%)")

    if glucose >= 180:
        st.markdown("<span style='color:red;font-weight:bold'>🔴 High Blood Sugar Detected</span>", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    ax.barh(["Risk"], [prob])
    ax.set_xlim(0, 100)
    st.pyplot(fig)

    st.session_state.history.append(f"Prediction → {risk} ({prob:.2f}%)")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Diabetes Prediction Report", ln=True)
    pdf.cell(0, 8, f"Risk: {risk}", ln=True)
    pdf.cell(0, 8, f"Probability: {prob:.2f}%", ln=True)
    pdf.multi_cell(0, 8, CAPTIONS[language]["disclaimer"])

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_pdf.name)

    with open(tmp_pdf.name, "rb") as f:
        st.download_button("📄 Download PDF", f.read(), "diabetes_report.pdf")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CHATBOT (ANSWER EVERYTHING) ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
q = st.text_input("💬 Ask anything about health, diabetes, hospitals, food, workout…")

if q:
    ql = q.lower()

    if "food" in ql or "diet" in ql:
        ans = (
            "🥗 Diabetic Diet:\n"
            "Vegetarian: vegetables, ragi, oats, dal\n"
            "Non-veg: boiled eggs, grilled fish/chicken\n\n"
            "🔴 If sugar is high: avoid sweets, white rice, junk food."
        )
    elif "medicine" in ql:
        ans = (
            "💊 Common diabetes medicines:\n"
            "- Metformin\n- Insulin\n- Glimepiride\n\n"
            "Consult doctor before use."
        )
    elif "gym" in ql or "exercise" in ql:
        ans = (
            "🏃 Exercise:\n"
            "- Walking 30 mins\n"
            "- Light gym\n"
            "- Yoga\n"
            "- Avoid heavy weights"
        )
    else:
        ans = ai_fallback(q)

    st.session_state.history.append(q)
    st.markdown(f"<div class='answer'>{ans}</div>", unsafe_allow_html=True)

    if st.button("🔊 Listen Answer"):
        speak(ans)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HISTORY ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🕘 History")
for h in st.session_state.history:
    st.write(h)

if st.button("🗑 Clear History"):
    st.session_state.history = []

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(f"""
<div class="footer">
<b>Kumar GK</b> | Student Academic Project<br>
{CAPTIONS[language]['disclaimer']}
</div>
""", unsafe_allow_html=True)

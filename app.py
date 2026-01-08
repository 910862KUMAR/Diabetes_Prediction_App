# ============================================================
# ADVANCED DIABETES AI ASSISTANT (RULE-BASED)
# Developed by Kumar GK
# Student Academic Project – ISE
# ============================================================

import streamlit as st
import numpy as np
import joblib
import tempfile
from fpdf import FPDF
from gtts import gTTS

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
.answer { background:#1f2937; padding:14px; border-radius:12px; white-space:pre-line; }
.footer { text-align:center; color:#9ca3af; font-size:14px; }
</style>
""", unsafe_allow_html=True)

# ---------------- LANGUAGE ----------------
language = st.selectbox("🌐 Language / ಭಾಷೆ / भाषा", ["English", "Kannada", "Hindi"])
LANG_CODE = {"English": "en", "Kannada": "kn", "Hindi": "hi"}

def translate(text):
    if language == "English":
        return text
    if language == "Kannada":
        return "👉 ಕನ್ನಡದಲ್ಲಿ ಮಾಹಿತಿ ನೀಡಲಾಗಿದೆ (ಡемо ಅನುವಾದ)\n\n" + text
    if language == "Hindi":
        return "👉 हिंदी में जानकारी दी गई है (डेमो अनुवाद)\n\n" + text

# ---------------- VOICE ----------------
def speak(text):
    tts = gTTS(text=text, lang=LANG_CODE[language])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)

# ---------------- HEADER ----------------
st.markdown("""
<div class="card">
<h1>Advanced Diabetes AI Assistant</h1>
<p><b>Developed by Kumar GK</b><br>
ISE Student – Academic Healthcare Project</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl"), joblib.load("scaler.pkl")

model, scaler = load_model()

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PREDICTION ----------------
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

    risk = "High Risk" if prob > 70 else "Medium Risk" if prob > 40 else "Low Risk"
    result = f"Risk Level: {risk} ({prob:.2f}%)"
    st.success(result)

    if glucose >= 180:
        st.markdown("<span style='color:red;font-weight:bold'>🔴 High Blood Sugar Detected</span>", unsafe_allow_html=True)

    st.session_state.history.append("Prediction → " + result)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Diabetes Prediction Report", ln=True)
    pdf.multi_cell(0, 8, result)
    pdf.multi_cell(0, 8, "Educational purpose only.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)

    with open(tmp.name, "rb") as f:
        st.download_button("📄 Download PDF", f.read(), "diabetes_report.pdf")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CHATBOT ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("💬 Health Chatbot (Rule-Based)")

q = st.text_input("Ask about hospital, food, medicine, workout, heart, kidney…")

ans = None
if q:
    ql = q.lower()

    # ---------------- HOSPITAL ----------------
    if "hospital" in ql:
        ans = (
            "🏥 Karnataka Hospitals:\n"
            "- Victoria Hospital – Bangalore (Government)\n"
            "- Jayadeva Heart Hospital – Bangalore\n"
            "- KR Hospital – Mysore\n"
            "- KIMS – Hubli\n"
            "- Siddaganga Hospital – Tumkur\n\n"
            "You can ask:\n"
            "• heart hospital\n• kidney hospital\n• government hospital"
        )

    # ---------------- FOOD ----------------
    elif "food" in ql or "diet" in ql:
        ans = (
            "🥗 Diabetic Diet Guide:\n\n"
            "👶 Children (Type 1):\n"
            "- Balanced meals\n- Fruits & vegetables\n- Avoid sweets\n\n"
            "🧑 Adults:\n"
            "- Ragi, oats, dal\n- Vegetables\n- Boiled eggs, grilled fish\n\n"
            "👴 Elderly:\n"
            "- Soft food\n- Low salt & sugar\n\n"
            "🔴 If sugar is high: avoid sweets, white rice, junk food"
        )

    # ---------------- MEDICINE ----------------
    elif "medicine" in ql:
        ans = (
            "💊 Diabetes Medicines (General Info):\n"
            "- Metformin\n- Glimepiride\n- Insulin\n\n"
            "⚠️ Medicine must be taken only after doctor consultation."
        )

    # ---------------- WORKOUT ----------------
    elif "exercise" in ql or "workout" in ql or "gym" in ql:
        ans = (
            "🏃 Exercise Guide:\n"
            "Children: walking, cycling\n"
            "Adults: brisk walk, light gym\n"
            "Elderly: yoga, stretching\n\n"
            "Avoid heavy weights if sugar is uncontrolled."
        )

    # ---------------- ORGANS ----------------
    elif "heart" in ql:
        ans = "❤️ Diabetes increases heart risk. Control sugar, BP and cholesterol."
    elif "kidney" in ql:
        ans = "🩺 Diabetes can damage kidneys. Regular tests required."
    elif "eye" in ql:
        ans = "👁️ Diabetes can affect eyesight. Annual eye checkup needed."
    else:
        ans = "Please ask about hospital, food, medicine, exercise, heart, kidney, eye."

    ans = translate(ans)
    st.session_state.history.append(q)
    st.markdown(f"<div class='answer'>{ans}</div>", unsafe_allow_html=True)

    if st.button("🔊 Listen Answer"):
        speak(ans)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HISTORY ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🕘 History")

for h in st.session_state.history:
    st.write("•", h)

if st.button("🗑 Clear History"):
    st.session_state.history = []

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
<b>Kumar GK</b> | ISE Student<br>
This application is developed only for academic project purposes.<br>
It does not replace professional medical advice.
</div>
""", unsafe_allow_html=True)

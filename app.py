# ============================================================
# ADVANCED DIABETES AI ASSISTANT (RULE-BASED)
# Developed by Kumar GK
# ISE Student – Academic Healthcare Project
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

# ---------------- COMPACT CSS ----------------
st.markdown("""
<style>
.stApp { background:#0b1220; color:#e5e7eb; }
.card { background:#111827; padding:12px; border-radius:14px; }
.answer { background:#1f2937; padding:10px; border-radius:10px; font-size:14px; white-space:pre-line; }
.footer { text-align:center; color:#9ca3af; font-size:12px; margin-top:6px; }
label { font-size:14px !important; }
input, select { height:36px !important; }
button { font-size:14px !important; padding:4px 12px !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- LANGUAGE ----------------
language = st.selectbox("🌐 Language / ಭಾಷೆ / भाषा", ["English", "Kannada", "Hindi"])
LANG_CODE = {"English": "en", "Kannada": "kn", "Hindi": "hi"}

def translate(text):
    if language == "English":
        return text
    if language == "Kannada":
        return "👉 ಕನ್ನಡದಲ್ಲಿ ಮಾಹಿತಿ (ಡೆಮೊ)\n\n" + text
    if language == "Hindi":
        return "👉 हिंदी में जानकारी (डेमो)\n\n" + text

# ---------------- VOICE ----------------
def speak(text):
    tts = gTTS(text=text, lang=LANG_CODE[language])
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)

# ---------------- HEADER ----------------
st.markdown("""
<div class="card">
<h2>Advanced Diabetes AI Assistant</h2>
<p><b>Kumar GK</b> – ISE Student<br>
Academic Healthcare Project</p>
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

# ---------------- TABS (ONE-PAGE LAYOUT) ----------------
tab1, tab2, tab3 = st.tabs(["🧪 Prediction", "💬 Chatbot", "🕘 History"])

# ========================================================
# TAB 1: DIABETES PREDICTION
# ========================================================
with tab1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Diabetes Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        glucose = st.number_input("Glucose Level", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 70)

    with col2:
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 60.0, 25.0)
        dpf = st.number_input("DPF", 0.0, 3.0, 0.5)

    pregnancies = 0
    if gender == "Female" and age >= 15:
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)

    if st.button("🔍 Predict Diabetes"):
        X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        Xs = scaler.transform(X)
        prob = model.predict_proba(Xs)[0][1] * 100

        risk = "High Risk" if prob > 70 else "Medium Risk" if prob > 40 else "Low Risk"
        result = f"Risk Level: {risk} ({prob:.2f}%)"
        st.success(result)

        if glucose >= 180:
            st.markdown("<span style='color:red;font-weight:bold'>🔴 High Blood Sugar</span>", unsafe_allow_html=True)

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

# ========================================================
# TAB 2: RULE-BASED CHATBOT
# ========================================================
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Health Chatbot")

    q = st.text_input(
        "Ask about hospital, food, medicine, workout, heart, kidney…",
        placeholder="Example: food for diabetic child"
    )

    if q:
        ql = q.lower()

        if "hospital" in ql:
            ans = (
                "🏥 Karnataka Hospitals:\n"
                "- Victoria Hospital – Bangalore\n"
                "- Jayadeva Heart Hospital – Bangalore\n"
                "- KR Hospital – Mysore\n"
                "- KIMS – Hubli\n"
                "- Siddaganga Hospital – Tumkur"
            )

        elif "food" in ql or "diet" in ql:
            ans = (
                "🥗 Diabetic Diet Guide:\n\n"
                "👶 Children:\n- Balanced meals\n- Fruits & vegetables\n\n"
                "🧑 Adults:\n- Ragi, oats, dal\n- Vegetables\n- Boiled eggs, grilled fish\n\n"
                "👴 Elderly:\n- Soft food\n- Low salt & sugar\n\n"
                "🔴 High sugar: avoid sweets & junk food"
            )

        elif "medicine" in ql:
            ans = (
                "💊 Common Diabetes Medicines:\n"
                "- Metformin\n- Glimepiride\n- Insulin\n\n"
                "⚠️ Doctor consultation required."
            )

        elif "exercise" in ql or "gym" in ql or "workout" in ql:
            ans = (
                "🏃 Exercise Guide:\n"
                "Children: walking, cycling\n"
                "Adults: brisk walk, light gym\n"
                "Elderly: yoga, stretching"
            )

        elif "heart" in ql:
            ans = "❤️ Diabetes increases heart disease risk."
        elif "kidney" in ql:
            ans = "🩺 Diabetes can damage kidneys."
        elif "eye" in ql:
            ans = "👁️ Diabetes can affect eyesight."
        else:
            ans = "Please ask about hospital, food, medicine, exercise, heart, kidney or eye."

        ans = translate(ans)
        st.session_state.history.append(q)
        st.markdown(f"<div class='answer'>{ans}</div>", unsafe_allow_html=True)

        if st.button("🔊 Listen Answer"):
            speak(ans)

    st.markdown("</div>", unsafe_allow_html=True)

# ========================================================
# TAB 3: HISTORY
# ========================================================
with tab3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("History")

    if st.session_state.history:
        for h in st.session_state.history:
            st.write("•", h)
    else:
        st.info("No history yet")

    if st.button("🗑 Clear History"):
        st.session_state.history = []

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
<b>Kumar GK</b> | ISE Student<br>
Educational purpose only • Not medical advice
</div>
""", unsafe_allow_html=True)

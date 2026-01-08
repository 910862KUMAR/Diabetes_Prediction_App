# ============================================================
# ADVANCED DIABETES AI ASSISTANT
# Developed by Kumar GK
# Student Academic Project
# ============================================================

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
from gtts import gTTS
import tempfile

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

# ---------------- LANGUAGE SELECT ----------------
language = st.selectbox(
    "🌐 Language / ಭಾಷೆ / भाषा",
    ["English", "Kannada", "Hindi"]
)

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

# ---------------- VOICE FUNCTION ----------------
def speak(text, lang):
    lang_code = {"English": "en", "Kannada": "kn", "Hindi": "hi"}[lang]
    tts = gTTS(text=text, lang=lang_code)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name)

# ---------------- KARNATAKA HOSPITAL DATABASE ----------------
KARNATAKA_HOSPITALS = {
    "Tumkur": {
        "government": {
            "diabetes": ["District Hospital Tumkur"],
            "heart": ["District Hospital Tumkur – Cardiology"],
            "kidney": ["District Hospital Tumkur – Nephrology"],
            "eye": ["Government Eye Hospital Tumkur"]
        },
        "private": {
            "diabetes": ["Siddaganga Hospital", "Sri Siddhartha Hospital"],
            "heart": ["Sri Siddhartha Heart Centre"],
            "kidney": ["Sri Siddhartha Nephrology"],
            "eye": ["Siddaganga Eye Hospital"]
        }
    },
    "Bangalore": {
        "government": {
            "diabetes": ["Victoria Hospital"],
            "heart": ["Jayadeva Institute of Cardiology"],
            "kidney": ["Victoria Hospital – Nephrology"],
            "eye": ["Minto Eye Hospital"]
        },
        "private": {
            "diabetes": ["Apollo Hospital", "Manipal Hospital"],
            "heart": ["Narayana Health", "Fortis Hospital"],
            "kidney": ["Manipal Nephrology"],
            "eye": ["Narayana Nethralaya"]
        }
    }
}

# ---------------- HELPERS ----------------
def detect_organ(q):
    q = q.lower()
    if "heart" in q: return "heart"
    if "kidney" in q: return "kidney"
    if "eye" in q: return "eye"
    return "diabetes"

def detect_type(q):
    if "government" in q or "govt" in q: return "government"
    if "private" in q: return "private"
    return None

# ---------------- STATE ----------------
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CHAT INPUT ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
query = st.text_input("💬 Ask about hospital, medicine, food, workout, gym...")

if query:
    q = query.lower()
    answer = ""

    # MEDICINE
    if "medicine" in q:
        answer = "💊 Common diabetes medicines:\n- Metformin\n- Insulin\n- Glimepiride\n\n" + CAPTIONS[language]["disclaimer"]

    # FOOD
    elif "food" in q or "diet" in q:
        answer = "🥗 Healthy diabetic diet:\n- Vegetables\n- Whole grains\n- Avoid sugar\n\n" + CAPTIONS[language]["disclaimer"]

    # WORKOUT
    elif "workout" in q or "gym" in q or "exercise" in q:
        answer = "🏃 Safe exercise:\n- Walking\n- Light gym\n- Yoga\n\n" + CAPTIONS[language]["disclaimer"]

    # HOSPITAL
    elif "hospital" in q:
        organ = detect_organ(q)
        htype = detect_type(q)
        for city in KARNATAKA_HOSPITALS:
            if city.lower() in q:
                answer = f"🏥 {organ.title()} hospitals in {city}:\n"
                if htype:
                    hs = KARNATAKA_HOSPITALS[city][htype][organ]
                else:
                    hs = (
                        KARNATAKA_HOSPITALS[city]["government"][organ] +
                        KARNATAKA_HOSPITALS[city]["private"][organ]
                    )
                for h in hs:
                    answer += f"- {h}\n"
                answer += "\n" + CAPTIONS[language]["disclaimer"]
                break
        if not answer:
            answer = "Please mention a Karnataka city like Tumkur or Bangalore."

    else:
        answer = "Please ask about hospital, medicine, food, or workout."

    st.session_state.last_answer = answer
    st.session_state.history.append(query)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ANSWER DISPLAY ----------------
if st.session_state.last_answer:
    st.markdown("<div class='answer'>", unsafe_allow_html=True)
    st.write(st.session_state.last_answer)
    if st.button("🔊 Listen Answer"):
        speak(st.session_state.last_answer, language)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- HISTORY ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🕘 History")
for i, h in enumerate(st.session_state.history):
    st.write(f"{i+1}. {h}")

if st.button("🗑 Clear History"):
    st.session_state.history = []
    st.session_state.last_answer = None
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(f"""
<div class="footer">
<b>Kumar GK</b> | Student Academic Project<br>
{CAPTIONS[language]['disclaimer']}
</div>
""", unsafe_allow_html=True)

# app.py — FINAL CLEAN VERSION (Robot + Voice + TTS + SHAP + PDF + Multilanguage)
import streamlit as st
import joblib
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
from gtts import gTTS
from fpdf import FPDF
from PIL import Image
from io import BytesIO
from openai import OpenAI

# ---------------- PAGE SETUP ---------------- #
st.set_page_config(
    page_title="AI Diabetes Prediction + Health Assistant",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CSS ---------------- #
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg,#0f1724 0%, #061122 100%);
    color: #e6eef8;
}
.card {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
.small { font-size: 0.9rem; color:#b9c6d9; }
.hero-title { font-size:32px; font-weight:700; margin:0; }
</style>
""", unsafe_allow_html=True)

# ------------- LOAD MODEL & SCALER ------------- #
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------- OPENAI CLIENT ------------- #
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY missing in Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------- SIDEBAR ---------------- #
with st.sidebar:
    st.title("Controls 💠")
    lang = st.selectbox("Assistant Language", ["English", "Kannada", "Hindi", "Tamil", "Telugu"])
    tts_enabled = st.checkbox("Enable Voice Reply (TTS)", value=True)
    st.markdown("---")
    st.info("Voice input supported.\nSHAP + PDF included.\nPlacement-ready project.")

# ------------- ROBOT AVATAR HEADER ------------- #
AVATAR_PATH = "/mnt/data/805721e3-bf3a-47d0-abc9-4b61d465afe9.png"

colA, colB = st.columns([0.15, 0.85])
with colA:
    try:
        img = Image.open(AVATAR_PATH).resize((110, 110))
        st.image(img)
    except:
        st.write("🤖")
with colB:
    st.markdown("""
    <div class="card">
        <h1 class="hero-title">AI Diabetes Prediction System</h1>
        <div class="small">Prediction • Diet & Exercise • Doctor • Multilanguage • SHAP • PDF • Voice Assistant</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- INPUT FORM ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Enter Patient Details")

c1, c2 = st.columns(2)
with c1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 2)
    glucose = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
with c2:
    insulin = st.number_input("Insulin Level", 0, 900, 85)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("DPF", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 33)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ---------------- #
def do_prediction():
    X = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    prob = float(model.predict_proba(Xs)[0][1])

    st.session_state["last_report"] = {
        "pregnancies": pregnancies, "glucose": glucose, "blood_pressure": blood_pressure,
        "skin": skin_thickness, "insulin": insulin, "bmi": bmi, "dpf": dpf,
        "age": age, "prediction": "High" if pred else "Low", "probability": prob
    }
    return pred, prob, Xs

prediction_result = None

if st.button("🔍 Predict"):
    prediction_result, probability, Xs = do_prediction()

    m1, m2, m3 = st.columns(3)
    m1.metric("Risk", "High" if prediction_result else "Low")
    m2.metric("Probability", f"{probability*100:.2f}%")
    m3.metric("BMI", f"{bmi:.1f}")

    if prediction_result == 1:
        st.error(f"⚠️ High Risk! ({probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({probability*100:.2f}%)")

# ---------------- SHAP EXPLAINABILITY ---------------- #
def generate_shap_images(Xs):
    paths = []
    try:
        import shap
        explainer = shap.Explainer(model, Xs)
        vals = explainer(Xs)

        # Bar plot
        plt.figure()
        shap.plots.bar(vals, show=False)
        p1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        paths.append(p1)
        plt.close()

        # Beeswarm
        plt.figure()
        shap.plots.beeswarm(vals, show=False)
        p2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        paths.append(p2)
        plt.close()

    except:
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            labels = ["Preg", "Gluc", "BP", "Skin", "Insulin", "BMI", "DPF", "Age"]
            idx = np.argsort(fi)[::-1]

            plt.figure()
            plt.bar([labels[i] for i in idx], fi[idx])
            plt.xticks(rotation=45)
            p3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            plt.savefig(p3, dpi=150, bbox_inches="tight")
            paths.append(p3)
            plt.close()

    return paths

# ---------------- CHATBOT + VOICE ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🤖 Health Assistant (Chat + Voice + Multi-language)")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": f"You reply only in {lang}. Provide friendly health guidance."}
    ]

# Display history
for msg in st.session_state.messages[1:]:
    st.markdown(f"<div class='small'><b>{msg['role']}:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Text input
user_text = st.text_input("Ask your question:", key="chat_text")
if user_text:
    st.session_state["pending_message"] = user_text

# Audio file
audio_file = st.file_uploader("Upload voice:", type=["wav", "mp3", "m4a"])

def transcribe_audio(data):
    try:
        resp = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=data
        )
        if hasattr(resp, "text"):
            return resp.text
    except:
        return None

# SEND BUTTON
if st.button("Send to Assistant"):
    final_msg = st.session_state.get("pending_message", None)

    if audio_file:
        trans = transcribe_audio(audio_file)
        if trans:
            final_msg = trans

    if final_msg:
        st.session_state.messages.append({"role": "user", "content": final_msg})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.messages
        )

        bot_msg = response.choices[0].message.content
        st.session_state.messages.append({"role": "assistant", "content": bot_msg})

        st.markdown(f"<div class='small'><b>Assistant:</b> {bot_msg}</div>", unsafe_allow_html=True)

        if tts_enabled:
            try:
                tts = gTTS(text=bot_msg, lang="en")
                mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
                tts.save(mp3)
                st.audio(open(mp3, "rb").read())
            except:
                st.warning("TTS failed.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- SHAP BUTTON ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Explainability (SHAP)")

if st.button("Generate SHAP"):
    if "last_report" not in st.session_state:
        st.warning("Run prediction first.")
    else:
        X = np.array([[st.session_state["last_report"][k] for k in 
                       ["pregnancies","glucose","blood_pressure","skin",
                        "insulin","bmi","dpf","age"]]])
        Xs = scaler.transform(X)
        imgs = generate_shap_images(Xs)
        st.session_state["shap_images"] = imgs
        for p in imgs:
            st.image(p)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PDF GENERATOR ---------------- #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Download PDF Report")

if st.button("Download PDF"):
    if "last_report" not in st.session_state:
        st.warning("Run prediction first.")
    else:
        rpt = st.session_state["last_report"]
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "AI Diabetes Report", 0, 1, "C")
        pdf.ln(5)

        pdf.set_font("Arial", size=10)
        for k, v in rpt.items():
            pdf.cell(0, 8, f"{k}: {v}", ln=True)

        if "shap_images" in st.session_state:
            for img in st.session_state["shap_images"]:
                pdf.add_page()
                try:
                    pdf.image(img, x=10, y=20, w=190)
                except:
                    pass

        out = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
        pdf.output(out)

        with open(out, "rb") as f:
            st.download_button("Download PDF", f.read(), file_name="diabetes_report.pdf")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #
st.markdown("<div class='small'>All features enabled ✔ Robot ✔ Voice ✔ PDF ✔ SHAP ✔ Multi-language</div>",
            unsafe_allow_html=True)

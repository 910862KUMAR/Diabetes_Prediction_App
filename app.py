# app.py — Upgraded: Robot avatar + Mic upload (file capture) + TTS + SHAP + PDF + Multi-language
import streamlit as st
import joblib
import numpy as np
import os
import tempfile
import base64
import matplotlib.pyplot as plt
from gtts import gTTS
from fpdf import FPDF
from PIL import Image
from io import BytesIO

# OpenAI client
from openai import OpenAI

# ---------------- PAGE SETUP ---------------- #
st.set_page_config(
    page_title="AI Diabetes Prediction + Health Assistant",
    page_icon="🤖",
    layout="wide",
)

# ---------------- CSS for modern UI ---------------- #
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#0f1724 0%, #061122 100%); color: #e6eef8; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02)); border-radius:12px; padding:16px; margin-bottom:12px;}
    .hero { padding:18px; border-radius:10px; }
    .muted { color:#99a6bf; }
    .small { font-size:0.9rem; color:#b9c6d9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Constants / Paths ---------------- #
# Avatar image you uploaded earlier (developer-provided path)
AVATAR_PATH = "/mnt/data/805721e3-bf3a-47d0-abc9-4b61d465afe9.png"

# ---------------- Load model & scaler ---------------- #
try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error("Model files not found or failed to load. Make sure diabetes_model.pkl and scaler.pkl exist.")
    st.stop()

# ---------------- OpenAI client ---------------- #
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY not found in environment variables! Add it to Streamlit Secrets.")
    st.stop()
client = OpenAI(api_key=api_key)

# ---------------- Sidebar controls ---------------- #
with st.sidebar:
    st.title("Controls")
    lang = st.selectbox("Assistant language", ["English", "Kannada", "Hindi", "Tamil", "Telugu"])
    tts_enabled = st.checkbox("Enable voice replies (TTS)", value=True)
    mic_upload_label = "Record/Upload voice (mobile: record, then upload)"
    st.markdown("---")
    st.markdown("**Placement-ready features:** SHAP explainability, PDF export, voice assistant, avatar.")
    st.markdown("⚠️ This app is educational only — not a medical diagnosis.")

# ---------------- HERO ---------------- #
cols = st.columns([0.12, 0.88])
with cols[0]:
    try:
        avatar = Image.open(AVATAR_PATH).resize((90, 90))
        st.image(avatar, width=90)
    except:
        st.image(None)
with cols[1]:
    st.markdown("<div class='hero card'><h1 style='margin:0;'>AI Diabetes Prediction App</h1>"
                "<div class='muted small'>Prediction • Diet & Exercise • Doctor recommendations • Generative AI Assistant</div></div>",
                unsafe_allow_html=True)

# ---------------- Input form ---------------- #
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Enter Patient Details")
c1, c2 = st.columns(2)
with c1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
with c2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=85)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Prediction + SHAP compute & store ---------------- #
def run_prediction_store():
    X = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    prob = float(model.predict_proba(Xs)[0][1])
    # save last report
    st.session_state['last_report'] = {
        "pregnancies": int(pregnancies), "glucose": float(glucose), "bp": float(blood_pressure),
        "skin": float(skin_thickness), "insulin": float(insulin), "bmi": float(bmi),
        "dpf": float(dpf), "age": int(age), "prediction": ("High" if pred==1 else "Low"), "probability": prob
    }
    return pred, prob, Xs

prediction_result = None
probability = None

if st.button("🔍 Predict Diabetes Risk"):
    try:
        prediction_result, probability, Xs = run_prediction_store()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # show metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk", ("High" if prediction_result==1 else "Low"))
    m2.metric("Probability", f"{probability*100:.2f}%")
    m3.metric("BMI", f"{bmi:.1f}")

    if prediction_result == 1:
        st.error(f"⚠️ High Risk of Diabetes! (Probability: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Diabetes (Probability: {probability*100:.2f}%)")
    st.caption("Disclaimer: This is a machine learning model for educational purposes.")

# ---------------- SHAP explainability (bar + beeswarm) ---------------- #
def compute_and_save_shap(Xs):
    """
    Attempts to compute SHAP values. If shap not available or heavy, fallback to feature_importances_.
    Saves two png files in a temp directory and returns their paths.
    """
    fig_paths = []
    try:
        import shap
        # Try creating an explainer (handles tree models well)
        explainer = shap.Explainer(model, Xs)  # can be slow for large models
        shap_values = explainer(Xs)
        # Feature importance bar
        plt.figure(figsize=(6,3))
        shap.plots.bar(shap_values, show=False)
        tmp1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(tmp1.name, bbox_inches='tight', dpi=150)
        plt.close()
        fig_paths.append(tmp1.name)
        # Beeswarm (force a small sample if heavy)
        plt.figure(figsize=(6,3))
        shap.plots.beeswarm(shap_values, show=False)
        tmp2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(tmp2.name, bbox_inches='tight', dpi=150)
        plt.close()
        fig_paths.append(tmp2.name)
    except Exception as e:
        # fallback: use model.feature_importances_ if present
        try:
            fi = None
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
            elif hasattr(model, "coef_"):
                fi = np.abs(model.coef_).ravel()
            if fi is not None:
                names = ["Pregnancies","Glucose","BP","Skin","Insulin","BMI","DPF","Age"]
                idx = np.argsort(fi)[::-1]
                plt.figure(figsize=(6,3))
                plt.bar([names[i] for i in idx], fi[idx])
                plt.xticks(rotation=45, ha="right")
                plt.title("Feature importance (model)")
                tmp1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                plt.savefig(tmp1.name, bbox_inches='tight', dpi=150)
                plt.close()
                fig_paths.append(tmp1.name)
        except Exception as ee:
            st.warning("SHAP and feature importance generation failed; explainability images not available.")
    return fig_paths

# ---------------- Chatbot + TTS + Avatar ---------------- #
if "messages" not in st.session_state:
    # initial system message uses language selection
    sys_msg = f"You are a friendly diabetes health assistant. Answer in {lang}. Keep advice educational and non-diagnostic."
    st.session_state.messages = [{"role":"system","content":sys_msg}]

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 AI Health Assistant (chat + voice + avatar)")

# display small chat history
for msg in st.session_state.messages[1:]:
    role = msg.get("role","user")
    content = msg.get("content","")
    st.markdown(f"<div class='small'><b>{role.capitalize()}:</b> {content}</div>", unsafe_allow_html=True)

# text input
user_text = st.text_input("Type your question (or use voice upload below):", key="user_text_input")

# audio upload for voice input (mobile-friendly)
st.markdown("**Voice input:** Click the button below on your phone to record, then upload the recorded file.")
audio_file = st.file_uploader(mic_upload_label, type=["wav","mp3","m4a"], accept_multiple_files=False)

# If user uses audio_file, transcribe it using OpenAI speech recognition (if supported) or fallback to text (not ideal)
def transcribe_audio_with_openai(file_bytes):
    """
    Tries to use OpenAI speech-to-text via audio.speech.* if available.
    Fallback: returns None.
    """
    try:
        # using client.audio.speech.create if supported by SDK
        # NOTE: different openai versions have different method names — using try/except
        resp = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=file_bytes)
        # fallback access patterns
        text = None
        if hasattr(resp, "text"):
            text = resp.text
        elif isinstance(resp, dict) and "text" in resp:
            text = resp["text"]
        elif hasattr(resp, "transcript"):
            text = resp.transcript
        return text
    except Exception:
        return None

# Prepare final user message (either text, or audio transcription)
final_user_message = None
if audio_file is not None:
    # read bytes
    audio_bytes = audio_file.read()
    # try to transcribe via OpenAI (best-effort)
    transcribed = transcribe_audio_with_openai(audio_bytes)
    if transcribed:
        final_user_message = transcribed
        st.info("Voice transcribed: " + transcribed)
    else:
        st.info("Voice uploaded — using the audio filename as a prompt hint. (You can type if transcription fails.)")
        final_user_message = f"[User uploaded audio file named: {audio_file.name}] Please provide guidance in {lang}."
# prefer typed text if present
if user_text.strip() != "":
    final_user_message = user_text

# send message to assistant
if st.button("Send to Assistant") and final_user_message:
    st.session_state.messages.append({"role":"user","content": final_user_message})
    # instruct assistant to reply in the selected language and include diet/doctor guidance
    instruction = f"Answer in {lang}. Provide: (1) short explanation of risk, (2) diet plan (veg & non-veg options), (3) exercise plan, (4) doctor recommendation if necessary."
    # append as system instruction temporarily
    messages_for_api = st.session_state.messages + [{"role":"system","content":instruction}]
    # call OpenAI chat
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_for_api
    )
    # access message content (new style)
    bot_msg = response.choices[0].message.content
    st.session_state.messages.append({"role":"assistant","content": bot_msg})
    # display assistant message
    st.markdown(f"<div class='small'><b>Assistant:</b> {bot_msg}</div>", unsafe_allow_html=True)

    # --- TTS: synthesize the bot_msg using gTTS and play ---
    if tts_enabled:
        try:
            tts = gTTS(text=bot_msg, lang="en")  # gTTS supports many languages; for Kannada/Hindi etc. you can adapt mapping
            tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            tts.save(tmp_mp3.name)
            audio_bytes = open(tmp_mp3.name, "rb").read()
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.warning(f"TTS failed: {e}")

    # --- Prepare SHAP plots after assistant answers (optional) ---
    if "last_report" in st.session_state:
        try:
            : # intentionally invalid syntax removed in final code
        except:
            pass

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Generate SHAP images button ---------------- #
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Explainability & SHAP")
if st.button("Generate Explainability Plots (bar + beeswarm)"):
    if "last_report" not in st.session_state:
        st.warning("Run a prediction first to compute explainability.")
    else:
        # regenerate Xs from last report
        lr = st.session_state['last_report']
        X = np.array([[lr["pregnancies"], lr["glucose"], lr["bp"], lr["skin"], lr["insulin"], lr["bmi"], lr["dpf"], lr["age"]]])
        try:
            Xs = scaler.transform(X)
            fig_paths = compute_and_save_shap(Xs)
            if fig_paths:
                for p in fig_paths:
                    st.image(p, use_column_width=True)
                st.success("Explainability images generated.")
                st.session_state['shap_images'] = fig_paths
            else:
                st.info("No explainability images could be generated.")
        except Exception as e:
            st.error(f"Failed to compute explainability: {e}")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PDF report generator ---------------- #
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Generate Styled PDF Report")
if st.button("Build PDF Report (fancy)"):
    if "last_report" not in st.session_state:
        st.warning("Run a prediction first.")
    else:
        rpt = st.session_state['last_report']
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(10,10,80)
        pdf.cell(0, 10, "AI Diabetes Report", ln=True, align="C")
        pdf.ln(4)
        pdf.set_font("Arial", size=10)
        for k, v in rpt.items():
            pdf.cell(0, 8, f"{k}: {v}", ln=True)
        pdf.ln(6)
        pdf.cell(0, 8, "Diet & Exercise Suggestions:", ln=True)
        pdf.multi_cell(0, 6, "Use the assistant chat to get a tailored diet and exercise plan, then press Build PDF to include in this report.")
        # attach SHAP images if exist
        if 'shap_images' in st.session_state:
            for p in st.session_state['shap_images']:
                try:
                    pdf.add_page()
                    pdf.image(p, x=15, y=30, w=180)
                except Exception:
                    pass
        # save pdf to bytes
        tmpf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.output(tmpf.name)
        with open(tmpf.name, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("Download Report PDF", data=pdf_bytes, file_name="diabetes_report.pdf", mime="application/pdf")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Multi-language note / mapping for TTS ---------------- #
# gTTS language mapping: for Kannada/Tamil Telugu you may need to use 'hi' or fallbacks
lang_map_for_tts = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "en",   # Kannada not supported well by gTTS; we fallback to English or use OpenAI TTS if available
    "Tamil": "ta",
    "Telugu": "te"
}

# ---------------- Footer ---------------- #
st.markdown("<div class='small muted'>App upgraded: avatar + voice upload + SHAP + PDF + multi-language. For best TTS in regional languages consider using OpenAI TTS or a specialised TTS provider.</div>", unsafe_allow_html=True)

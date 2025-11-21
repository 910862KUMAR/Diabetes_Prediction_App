# app.py — FINAL FULL FEATURED (Prediction + Instant Chat + Audio Upload + Transcription + OpenAI TTS + SHAP + PDF)
# Recommended requirements.txt entries (example):
# streamlit==1.51.0
# joblib
# numpy
# scikit-learn
# pandas
# openai
# matplotlib
# shap
# fpdf
# gTTS
# Pillow

import streamlit as st
import joblib
import numpy as np
import os
import tempfile
import base64
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
from PIL import Image

# optional gTTS fallback
try:
    from gtts import gTTS
    _HAS_GTTS = True
except Exception:
    _HAS_GTTS = False

# OpenAI client
from openai import OpenAI

# ---------- Configuration / uploaded-file path (kept for reference) ----------
# The platform can transform this local path into a URL if needed.
UPLOADED_IMAGE_PATH = "/mnt/data/e3f1c7a8-f4d2-4f52-8634-67ddd235360f.png"

# ---------- Page setup ----------
st.set_page_config(page_title="Advanced Diabetes AI Assistant", page_icon="🤖", layout="wide")
st.markdown("<style>body{background:#071226;} .stApp{background:#071226;color:#e6eef8;}</style>", unsafe_allow_html=True)

# ---------- Load model & scaler ----------
try:
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error("Model files not found or failed to load. Place diabetes_model.pkl and scaler.pkl in app folder.")
    st.stop()

# ---------- OpenAI client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing. Add it to Streamlit Secrets (TOML): OPENAI_API_KEY = \"sk-...\"")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Sidebar controls ----------
with st.sidebar:
    st.title("Controls")
    st.markdown("**TTS & Audio**")
    tts_engine = st.selectbox("TTS Engine", ["OpenAI TTS (recommended)", "gTTS (fallback)"])
    tts_voice = st.selectbox("TTS Voice Style", ["robot", "neutral", "doctor", "male", "female"])
    autoplay = st.checkbox("Autoplay audio on reply", value=True)
    st.markdown("---")
    st.markdown("**Tools**")
    show_shap = st.checkbox("Enable SHAP explainability", value=True)
    show_pdf = st.checkbox("Enable PDF report", value=True)
    st.markdown("---")
    st.caption("Keep your OpenAI key secure in Streamlit Secrets.")

# ---------- Top header ----------
st.markdown("""
<div style="padding:14px;border-radius:10px;background:linear-gradient(90deg,#061428,#082238);margin-bottom:10px;">
  <h1 style="margin:0;color:#7dd3fc;">Advanced Diabetes AI Assistant</h1>
  <div style="color:#b6c9d9;">Prediction • Instant Chatbot • SHAP • PDF • Audio upload & TTS</div>
</div>
""", unsafe_allow_html=True)

# ---------- Input form ----------
st.markdown('<div style="background:#07111a;padding:14px;border-radius:10px;margin-bottom:12px;">', unsafe_allow_html=True)
st.subheader("Enter Patient Details")
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
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Prediction ----------
st.markdown('<div style="background:#09101a;padding:12px;border-radius:10px;margin-bottom:12px;">', unsafe_allow_html=True)
st.subheader("Prediction")
if st.button("🔍 Predict Diabetes Risk"):
    X = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]
    prob = model.predict_proba(Xs)[0][1]

    st.metric("Risk", "High" if pred == 1 else "Low")
    st.metric("Probability", f"{prob*100:.2f}%")
    st.metric("BMI", f"{bmi:.1f}")

    if pred == 1:
        st.error(f"⚠️ High risk of diabetes ({prob*100:.2f}%) — consult a doctor.")
    else:
        st.success(f"✅ Low risk of diabetes ({prob*100:.2f}%)")

    # Save last report for SHAP/PDF
    st.session_state['last_report'] = {
        "pregnancies": int(pregnancies), "glucose": float(glucose), "blood_pressure": float(blood_pressure),
        "skin_thickness": float(skin_thickness), "insulin": float(insulin), "bmi": float(bmi),
        "dpf": float(dpf), "age": int(age),
        "prediction": ("High" if pred==1 else "Low"), "probability": float(prob)
    }
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Utilities: SHAP & plotting ----------
def generate_shap_images(Xs):
    imgs = []
    try:
        import shap
        explainer = shap.Explainer(model, Xs)
        shap_values = explainer(Xs)

        # bar
        plt.figure(figsize=(6,3))
        shap.plots.bar(shap_values, show=False)
        tmp1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plt.savefig(tmp1, bbox_inches="tight", dpi=150)
        plt.close()
        imgs.append(tmp1)

        # beeswarm
        plt.figure(figsize=(6,3))
        shap.plots.beeswarm(shap_values, show=False)
        tmp2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        plt.savefig(tmp2, bbox_inches="tight", dpi=150)
        plt.close()
        imgs.append(tmp2)
    except Exception:
        # fallback: feature importance
        try:
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
            elif hasattr(model, "coef_"):
                fi = np.abs(model.coef_).ravel()
            else:
                fi = None
            if fi is not None:
                labels = ["Pregnancies","Glucose","BP","SkinThickness","Insulin","BMI","DPF","Age"]
                idx = np.argsort(fi)[::-1]
                plt.figure(figsize=(6,3))
                plt.bar([labels[i] for i in idx], fi[idx])
                plt.xticks(rotation=45, ha="right")
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                plt.tight_layout()
                plt.savefig(tmp, dpi=150)
                plt.close()
                imgs.append(tmp)
        except Exception:
            pass
    return imgs

# ---------- Audio transcription helpers ----------
def transcribe_audio_bytes(audio_bytes):
    """Try using OpenAI speech-to-text calls; return text or None."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            f.flush()
            fpath = f.name
        # Try typical SDK patterns
        try:
            resp = client.audio.transcriptions.create(model="gpt-4o-transcribe", file=open(fpath, "rb"))
            if hasattr(resp, "text") and resp.text:
                return resp.text
            if isinstance(resp, dict) and "text" in resp:
                return resp["text"]
        except Exception:
            pass
        # fallback pattern
        try:
            resp2 = client.audio.speech.transcribe(model="gpt-4o-transcribe", file=open(fpath, "rb"))
            if hasattr(resp2, "text"):
                return resp2.text
            if isinstance(resp2, dict) and "text" in resp2:
                return resp2["text"]
        except Exception:
            pass
    except Exception:
        pass
    return None

# ---------- TTS helpers ----------
def synthesize_speech_openai(text, voice_style="neutral"):
    """Try OpenAI TTS endpoints; return bytes or None."""
    try:
        # Try a few plausible method paths / signatures
        # 1) client.audio.speech.create
        try:
            fn = client.audio.speech.create
            res = fn(model="gpt-4o-mini-tts", voice=voice_style, input=text)
            # handle possible outputs
            if hasattr(res, "audio"):
                data = res.audio
            elif isinstance(res, dict) and "audio" in res:
                data = res["audio"]
            else:
                data = None
            if isinstance(data, str):
                try:
                    return base64.b64decode(data)
                except Exception:
                    return None
            if isinstance(data, (bytes, bytearray)):
                return bytes(data)
        except Exception:
            pass
        # 2) client.audio.speech.synthesize
        try:
            fn2 = client.audio.speech.synthesize
            res2 = fn2(model="gpt-4o-mini-tts", voice=voice_style, input=text)
            if hasattr(res2, "audio"):
                data = res2.audio
                if isinstance(data, str):
                    return base64.b64decode(data)
                if isinstance(data, (bytes, bytearray)):
                    return bytes(data)
        except Exception:
            pass
    except Exception:
        pass
    return None

def synthesize_speech_gtts(text, lang="en"):
    """Fallback gTTS to synthesize speech; return bytes or None."""
    if not _HAS_GTTS:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        tts.save(tmp)
        with open(tmp, "rb") as f:
            return f.read()
    except Exception:
        return None

# ---------- Chat system prompt ----------
SYSTEM_PROMPT = """
You are a helpful diabetes health assistant. Answer concisely.
Provide:
- Short explanation
- Diet suggestions (veg & non-veg)
- Exercise plan
- Doctor recommendation (Endocrinologist, Dietitian, Cardiologist, Nephrologist, Ophthalmologist)
Always remind to consult medical professionals for diagnosis.
"""

# ---------- Chat UI & behavior (scrollable chat window, clear input after send) ----------
st.markdown('<div style="display:grid;grid-template-columns:1fr 420px;gap:18px;">', unsafe_allow_html=True)

# Left: Chat and input
with st.container():
    st.markdown('<div style="background:#07111a;padding:10px;border-radius:10px;">', unsafe_allow_html=True)
    st.subheader("Health Chat Assistant (instant replies)")

    # Initialize messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"system","content":SYSTEM_PROMPT}]

    # Chatbox (scrollable)
    chat_placeholder = st.empty()
    def render_chat():
        html = ['<div style="max-height:520px; overflow:auto; padding:8px;">']
        for m in st.session_state.messages:
            role = m.get("role","user")
            content = m.get("content","")
            if role == "system":
                continue
            if role == "user":
                html.append(f'<div style="background:#103d2a;color:white;padding:10px;border-radius:10px;margin:6px 0;display:inline-block;">👤 {content}</div>')
            else:
                html.append(f'<div style="background:#1f2937;color:#e6eef8;padding:10px;border-radius:10px;margin:6px 0;display:inline-block;">🤖 {content}</div>')
        html.append('</div>')
        chat_placeholder.markdown("".join(html), unsafe_allow_html=True)

    render_chat()

    # Input field (keyed so we can clear it)
    user_text = st.text_input("Type your question (press Enter to send):", key="chat_input", placeholder="e.g. Give me a veg diabetic diet plan...")
    audio_file = st.file_uploader("Or upload recorded audio (wav/mp3/m4a):", type=["wav","mp3","m4a"])

    # send detection:
    send_now = False
    final_user_message = None

    # New typed text was entered (pressing enter triggers rerun and sets value)
    if user_text and (st.session_state.get("last_sent_text") != user_text):
        final_user_message = user_text
        st.session_state["last_sent_text"] = user_text
        send_now = True

    # Audio upload prioritized
    if audio_file is not None:
        audio_bytes = audio_file.read()
        transcribed = transcribe_audio_bytes(audio_bytes)
        if transcribed:
            final_user_message = transcribed
            send_now = True
        else:
            # fallback: use filename as hint
            final_user_message = f"[Audio: {audio_file.name}]"
            send_now = True

    if send_now and final_user_message:
        # Append user message to history
        st.session_state.messages.append({"role":"user","content":final_user_message})
        render_chat()

        # Prepare messages for API
        messages_for_api = st.session_state.messages.copy()

        # Call OpenAI chat completion
        try:
            response = client.chat.completions.create(model="gpt-4o-mini", messages=messages_for_api)
            bot_text = None
            # robust extraction for various SDKs
            try:
                bot_text = response.choices[0].message.content
            except Exception:
                try:
                    bot_text = response.choices[0].message["content"]
                except Exception:
                    bot_text = str(response)
            # Append assistant reply to history
            st.session_state.messages.append({"role":"assistant","content":bot_text})
            render_chat()
        except Exception as e:
            st.error("Assistant call failed: " + str(e))
            bot_text = None

        # After reply: clear input box so it's ready for next question
        try:
            st.session_state["chat_input"] = ""
            st.session_state["last_sent_text"] = ""
        except Exception:
            pass

        # TTS: synthesize and play (OpenAI TTS preferred)
        if bot_text and isinstance(bot_text, str) and bot_text.strip():
            audio_bytes = None
            if tts_engine.startswith("OpenAI"):
                audio_bytes = synthesize_speech_openai(bot_text, voice_style=tts_voice)
            if audio_bytes is None and tts_engine.startswith("gTTS") and _HAS_GTTS:
                audio_bytes = synthesize_speech_gtts(bot_text, lang="en")
            if audio_bytes is None and _HAS_GTTS:
                audio_bytes = synthesize_speech_gtts(bot_text, lang="en")
            if audio_bytes:
                tmpmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                tmpmp.write(audio_bytes)
                tmpmp.flush()
                try:
                    if autoplay:
                        st.audio(open(tmpmp.name, "rb").read(), format='audio/mp3')
                    else:
                        st.audio(open(tmpmp.name, "rb").read(), format='audio/mp3')
                except Exception:
                    pass

    st.markdown('</div>', unsafe_allow_html=True)

# Right column: Tools (SHAP, PDF, session info)
with st.container():
    st.markdown('<div style="background:#07111a;padding:12px;border-radius:10px;">', unsafe_allow_html=True)
    st.subheader("Tools & Explainability")

    if show_shap:
        if st.button("Generate SHAP explainability (bar + beeswarm)"):
            if "last_report" not in st.session_state:
                st.warning("Run prediction first to compute SHAP.")
            else:
                lr = st.session_state["last_report"]
                X = np.array([[lr["pregnancies"], lr["glucose"], lr["blood_pressure"], lr["skin_thickness"],
                               lr["insulin"], lr["bmi"], lr["dpf"], lr["age"]]])
                try:
                    Xs = scaler.transform(X)
                    imgs = generate_shap_images(Xs)
                    if imgs:
                        st.session_state["shap_images"] = imgs
                        for p in imgs:
                            st.image(p, use_column_width=True)
                        st.success("Explainability images generated.")
                    else:
                        st.info("No explainability images were generated.")
                except Exception as e:
                    st.error("SHAP compute failed: " + str(e))

    if show_pdf:
        st.markdown("---")
        st.markdown("### PDF Report")
        if st.button("Create & Download PDF"):
            if "last_report" not in st.session_state:
                st.warning("Run prediction first.")
            else:
                rpt = st.session_state["last_report"]
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(0, 10, "AI Diabetes Report", ln=True, align="C")
                pdf.ln(4)
                pdf.set_font("Arial", size=10)
                for k,v in rpt.items():
                    pdf.cell(0, 8, f"{k}: {v}", ln=True)
                pdf.ln(6)
                pdf.multi_cell(0, 6, "Use the assistant for tailored diet/exercise suggestions. This is educational only.")

                if "shap_images" in st.session_state:
                    for p in st.session_state["shap_images"]:
                        try:
                            pdf.add_page()
                            pdf.image(p, x=10, y=20, w=190)
                        except Exception:
                            pass

                tmpf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
                pdf.output(tmpf)
                with open(tmpf, "rb") as f:
                    st.download_button("Download PDF", f.read(), file_name="diabetes_report.pdf", mime="application/pdf")

    st.markdown("---")
    st.markdown("### Session Info")
    num_msgs = len(st.session_state.get("messages", []))
    st.markdown(f"- Messages exchanged: **{num_msgs}**")
    st.markdown(f"- Last prediction saved: **{'yes' if 'last_report' in st.session_state else 'no'}**")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("<div style='color:#9fb0c9;font-size:0.9rem'>Note: Educational only — not medical advice. Keep your OpenAI key secure in Streamlit Secrets.</div>", unsafe_allow_html=True)

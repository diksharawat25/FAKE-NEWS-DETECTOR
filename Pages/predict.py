import streamlit as st
from streamlit_lottie import st_lottie
import json
import PyPDF2
import pickle
from gtts import gTTS
import os
import base64

# ✅ Set page config FIRST
st.set_page_config(page_title="Prediction", layout="wide")

# Load and apply custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load animations
def load_animation(path):
    with open(path, "r") as f:
        return json.load(f)

real_anim = load_animation("real.json")
fake_anim = load_animation("fake.json")

# Load model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Main UI
st.title("📂 Upload News File (PDF/TXT)")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file:
    text = ""
    if uploaded_file.type == "application/pdf":
        pdf = PyPDF2.PdfReader(uploaded_file)
        for page in pdf.pages:
            text += page.extract_text()
    else:
        text = uploaded_file.read().decode("utf-8")

    if text.strip():
        st.subheader("📜 Extracted Text:")
        st.text_area("News Content", text, height=200)

        if st.button("🔍 Predict"):
            vec = vectorizer.transform([text])
            result = model.predict(vec)[0]
            confidence_score = model.decision_function(vec)[0]
            confidence_percent = min(round(abs(confidence_score) * 10, 2), 100)

            if result == 1:
               st.error(f"🚨 FAKE NEWS DETECTED ({confidence_percent}% confidence)")
               st_lottie(fake_anim, height=200)
               output = "This news is fake."
            else:
               st.success(f"✅ REAL NEWS ({confidence_percent}% confidence)")
               st_lottie(real_anim, height=200)
               output = "This news is real."


            # Text-to-Speech
            tts = gTTS(output)
            tts.save("result.mp3")
            
            # Embed autoplay audio
            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{base64.b64encode(open("result.mp3", "rb").read()).decode()}" type="audio/mp3">
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            os.remove("result.mp3")
    else:
        st.warning("No text found in the file.")

# --------------------------
# ✍️ User Input Text Section
# --------------------------

st.markdown("---")
st.markdown("## ✍️ Try It Yourself!")
st.markdown("Type or paste any news content below and check if it's **Real** or **Fake**.")

user_input = st.text_area("📰 Enter news text here:", height=150)

if st.button("🧠 Analyze Typed News"):
    if user_input.strip() != "":
        vec = vectorizer.transform([user_input])
        result = model.predict(vec)[0]

        if result == 1:
            st.error("🚨 FAKE NEWS DETECTED")
            st_lottie(fake_anim, height=200)
            output = "This news is fake."
        else:
            st.success("✅ REAL NEWS")
            st_lottie(real_anim, height=200)
            output = "This news is real."

        # Text-to-Speech with Auto-Play
        tts = gTTS(output)
        tts.save("result.mp3")
        
        # Embed autoplay audio
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{base64.b64encode(open("result.mp3", "rb").read()).decode()}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        os.remove("result.mp3")

    else:
        st.warning("⚠️ Please enter some text to check.")


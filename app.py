import streamlit as st

# Page config - MUST be first Streamlit command
st.set_page_config(page_title="Fake News Detector", layout="centered")

from streamlit_lottie import st_lottie
import json

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load animation
def load_animation(path):
    with open(path, "r") as f:
        return json.load(f)

home_animation = load_animation("home.json")

st.title("📰 Fake News Detector")
st_lottie(home_animation, height=300)

st.markdown("Welcome to the Fake News Detector App. Upload news files and check whether it's **Real** or **Fake**.")

if st.button("🚀 Get Started"):
    st.switch_page("pages/predict.py")

# python -m streamlit run app.py
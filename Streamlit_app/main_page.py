import streamlit as st

# Title and description
st.set_page_config(page_title="Sign Language to English", page_icon="🤟", layout="wide")
st.title("Sign Language to English Text Conversion")
st.write("This project converts sign hand gestures to English text.")
# Define the pages
Model_Stats = st.Page("model_stats.py", title="Model_Stats", icon="📈")
Webcam = st.Page("app.py", title="Webcam", icon="📷")
# Set up navigation
pg = st.navigation([Webcam, Model_Stats])

# Run the selected page
pg.run()
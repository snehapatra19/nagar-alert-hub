import streamlit as st
import pickle
import re
import pandas as pd
from datetime import datetime

# Load model and vectorizer
model = pickle.load(open("issue_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

st.title("Nagar Alert Hub")
st.subheader("AI-Based Civic Issue Classification")

user_input = st.text_area("Describe the civic issue:")

if st.button("Analyze Issue"):

    if user_input.strip() == "":
        st.warning("Please enter an issue description")

    else:
        cleaned = clean_text(user_input)

        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        confidence = model.predict_proba(vec).max() * 100

        st.success(f"Predicted Issue Category: {prediction.upper()}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Save complaint automatically
        log = pd.DataFrame({
            "timestamp":[datetime.now()],
            "issue_text":[user_input],
            "predicted_category":[prediction],
            "confidence":[confidence]
        })

        try:
            existing = pd.read_csv("complaint_log.csv")
            updated = pd.concat([existing, log], ignore_index=True)
            updated.to_csv("complaint_log.csv", index=False)
        except:
            log.to_csv("complaint_log.csv", index=False)

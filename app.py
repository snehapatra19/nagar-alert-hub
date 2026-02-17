import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# HuggingFace model name
MODEL_NAME = "snehapatra1910/nagar-alert-model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Labels
labels = ["Power", "Road", "Garbage", "Water"]

# Prediction function
def predict_issue(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits).item()
    return labels[pred]

# UI
st.title("Nagar Alert Hub")
st.subheader("AI Based Civic Issue Classification")

text = st.text_area("Describe the civic issue")

if st.button("Analyze Issue"):
    if text:
        category = predict_issue(text)
        st.success(f"Predicted Issue Category: {category}")
    else:
        st.warning("Please enter an issue description")

import streamlit as st
import pickle

# Load saved model and vectorizer
model = pickle.load(open("issue_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Nagar Alert Hub", layout="centered")

st.title("Nagar Alert Hub")
st.subheader("AI Based Civic Issue Classification")

user_input = st.text_area("Describe the civic issue")

if st.button("Analyze Issue"):
    if user_input.strip() == "":
        st.warning("Please enter a description.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]

        st.success(f"Predicted Category: {prediction}")

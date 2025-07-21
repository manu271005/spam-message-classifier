import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# App title
st.title("📧 Spam Detection App")
st.write("Enter a message to check if it's Spam or Ham.")

# Input text
message = st.text_area("Enter your message here:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform and predict
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        probability = model.predict_proba(message_vec)[0][prediction]

        label = "Spam" if prediction == 1 else "Ham"
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {probability*100:.2f}%")

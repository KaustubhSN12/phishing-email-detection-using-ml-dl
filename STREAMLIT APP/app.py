
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Email Phishing Detection System", layout="wide")
st.title("📧 Email Phishing Detection System")
st.markdown("""
This interactive app detects whether an email is **Phishing** or **Legitimate** based on its content.
It uses text analysis and machine learning for cybersecurity awareness and prevention.
""")

# ------------------------------
# Load Dataset
# ------------------------------
st.header("📁 Dataset Overview")

df = None  # Initialize df to None
try:
    df = pd.read_csv("/content/drive/MyDrive/SEM_3_Project/augmented_dataset.csv")
    st.success("Default dataset loaded successfully!")
    st.write(df.head())
except FileNotFoundError:
    st.warning("Default dataset not found! Please upload one below.")

uploaded_file = st.file_uploader("Upload your email dataset (CSV)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded successfully!")
    st.write(df.head())

# ------------------------------
# Keyword Analysis
# ------------------------------
if df is not None and 'Email_Content' in df.columns:
    st.subheader("🔍 Phishing Keyword Frequency Analysis")

    phishing_words = [
        "verify", "account", "password", "urgent", "click", "update",
        "bank", "login", "secure", "alert", "confirm", "unsubscribe", "information"
    ]
    freq = [df['Email_Content'].str.contains(word, case=False, na=False).sum() for word in phishing_words]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(phishing_words, freq, color='teal')
    ax.set_title("Common Phishing-related Word Frequency")
    ax.set_xlabel("Keywords")
    ax.set_ylabel("Count")
    st.pyplot(fig)
elif df is None:
    st.warning("Dataset not loaded. Cannot perform keyword analysis.")
else:
    st.warning("Column 'Email_Content' not found in dataset.")

# ------------------------------
# Load Model and TF-IDF Vectorizer
# ------------------------------
st.header("🧠 Model Loading")

model_path = "/content/drive/MyDrive/SEM_3_Project/logistic_regression_model.pkl"
tfidf_path = "/content/drive/MyDrive/SEM_3_Project/tfidf_vectorizer.pkl"
#final_model = "/content/drive/MyDrive/SEM_3_Project/final_model.pkl"


# Load model
try:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
except:
    st.error("⚠️ Could not load model. Please check the path or upload file.")
    model = None

# Load TF-IDF vectorizer
try:
    tfidf_vectorizer = joblib.load(tfidf_path)
    st.success("✅ TF-IDF Vectorizer loaded successfully! along with Bert")
except:
    st.error("⚠️ Could not load TF-IDF vectorizer. Please check the path or upload file.")
    tfidf_vectorizer = None

# ------------------------------
# Prediction Functions
# ------------------------------
def simple_predict(text):
    """Basic keyword-based rule if ML model not loaded"""
    phishy_terms = ["verify", "password", "urgent", "account", "click", "bank", "secure"]
    matches = [w for w in phishy_terms if re.search(w, text, re.IGNORECASE)]
    return "Phishing" if len(matches) > 0 else "Legitimate"

def predict_email(email_content):
    """Predict using ML model or fallback to rule-based"""
    if model is None or tfidf_vectorizer is None:
        return simple_predict(email_content)

    try:
        # Step 1: TF-IDF vectorization
        tfidf_features = tfidf_vectorizer.transform([email_content])

        # Step 2: Compute additional feature (spam word count)
        spam_words = ["win", "free", "prize", "offer", "money", "urgent", "lottery", "click", "account"]
        spam_count = sum(word in email_content.lower() for word in spam_words)
        spam_feature = csr_matrix(np.array([[spam_count]]))

        # Step 3: Combine TF-IDF + spam count (must match training dimension)
        combined_features = hstack([tfidf_features, spam_feature])

        # Step 4: Predict
        prediction = model.predict(combined_features)
        result = "🚨 Phishing Email Detected" if prediction[0] == 1 else "✅ Legitimate Email"
        return result

    except Exception as e:
        return f"Error during model prediction: {e}"

# ------------------------------
# Text Input & Prediction Section
# ------------------------------
st.header("✉️ Email Content Prediction")

email_input = st.text_area("Enter email content to analyze:")

if st.button("🔎 Predict"):
    if email_input.strip() == "":
        st.warning("Please enter an email text.")
    else:
        result = predict_email(email_input)
        st.success(f"🧾 Prediction: **{result}**")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("Developed by **Kaustubh Narayankar** | MSc Data Science | SIES College")

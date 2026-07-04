import os
import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------------------
# Fix for system issues
# -----------------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(
    page_title="Shop Ease Sentiment AI",
    page_icon="🛍️",
    layout="centered"
)

# -----------------------------------------
# Title / Intro
# -----------------------------------------
st.markdown(
    """
    # 🛍️ Shop Ease
    ### Customer Sentiment Analysis App

    This app uses a fine-tuned **DistilBERT model** to classify customer reviews into:

    - 😠 Negative
    - 😐 Neutral
    - 😊 Positive

    It helps understand customer feedback quickly and easily.
    """
)

st.divider()

# -----------------------------------------
# Load Model
# -----------------------------------------
@st.cache_resource
def load_model():
    MODEL_NAME = "yemiabod/customer-sentiment-distilbert"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------------------
# Labels
# -----------------------------------------
labels = {
    0: "Negative 😠",
    1: "Neutral 😐",
    2: "Positive 😊"
}

# -----------------------------------------
# Model Info
# -----------------------------------------
st.subheader("📊 Model Performance")

st.write("""
- Accuracy: 90.93%
- ROC-AUC: 95.72%
- Weighted F1-Score: 98.11%
- Precision: 88.30%
- Recall: 90.93%
""")

st.divider()

# -----------------------------------------
# Session State
# -----------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None

# -----------------------------------------
# Input Section
# -----------------------------------------
st.header("Enter a Customer Review")

review = st.text_area("Write your review below")

predict_btn = st.button("Predict Sentiment")

# -----------------------------------------
# Prediction
# -----------------------------------------
if predict_btn:

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities).item()
        confidence = probabilities[0][prediction].item()

        st.session_state.prediction = prediction

        st.success(f"Prediction: {labels[prediction]}")
        st.write(f"Confidence: {confidence:.2%}")

        # Probability chart
        probs = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Probability": probabilities.numpy()[0]
        })

        st.subheader("Prediction Probabilities")
        st.bar_chart(probs.set_index("Sentiment"))

# -----------------------------------------
# Footer
# -----------------------------------------
st.divider()
st.caption("Built with DistilBERT | Streamlit App | Shop Ease")
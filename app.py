import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
# Title
# -----------------------------------------

st.title("🛍 Customer Sentiment Analysis")

st.write(
"""
This application predicts customer sentiment using a fine-tuned DistilBERT model.
"""
)

# -----------------------------------------
# Model Summary
# -----------------------------------------

st.subheader("Model Summary")

st.write("""
- Model: DistilBERT
- Accuracy: **90.93%**
- ROC-AUC: **95.72%**
- Weighted F1-Score: **98.11%**
- Weighted Precision: **88.30%%**
- Weighted Recall: **90.93%**
- Best performing model among all tested models.
""")

st.divider()

# -----------------------------------------
# User Input
# -----------------------------------------

review = st.text_area("Enter a customer review")

# -----------------------------------------
# Prediction
# -----------------------------------------

if st.button("Predict Sentiment"):

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

        st.success(f"Prediction: {labels[prediction]}")

        st.write(f"Confidence: {confidence:.2%}")

        # Probability table

        probs = pd.DataFrame({

            "Sentiment": ["Negative", "Neutral", "Positive"],

            "Probability": probabilities.numpy()[0]

        })

        st.subheader("Prediction Probabilities")

        st.bar_chart(probs.set_index("Sentiment"))
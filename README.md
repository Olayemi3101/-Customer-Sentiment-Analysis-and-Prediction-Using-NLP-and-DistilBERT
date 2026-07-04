# Customer Sentiment Analysis and Prediction Using NLP and DistilBERT

## Overview

This project develops an end-to-end Natural Language Processing (NLP) solution to automatically classify customer reviews into **Negative**, **Neutral**, and **Positive** sentiments. The project compares traditional machine learning models with a transformer-based deep learning model (DistilBERT) to identify the best-performing approach for customer sentiment prediction.

The solution also extracts key business insights from customer feedback and presents the results through an interactive Power BI dashboard and a Streamlit web application.

---

## Business Introduction

ShopEase Europe is a large online shopping company that receives thousands of customer reviews across its products and services. These reviews contain valuable insights into customer satisfaction, product quality, delivery performance, and customer service. However, manually analysing large volumes of feedback is inefficient and time-consuming. This project leverages Natural Language Processing (NLP), Machine Learning, and Deep Learning to automatically classify customer sentiment, identify key satisfaction drivers and complaints, and provide actionable insights to support data-driven business decisions and improve customer experience.

---

## Business Problem

Customer reviews provide valuable insights into customer satisfaction, product quality, and service performance. However, manually analysing thousands of reviews is time-consuming and inefficient.

This project helps ShopEase Europe to:

- Analyse customer sentiment
- Identify common customer complaints
- Discover sentiment drivers
- Support data-driven business decisions
- Improve customer experience

---

## Project Objectives

- Clean and preprocess customer review data
- Perform Exploratory Data Analysis (EDA)
- Identify sentiment drivers using NLP
- Train baseline machine learning models
- Fine-tune a DistilBERT model
- Compare model performance
- Build an interactive Streamlit application
- Develop a Power BI dashboard
- Generate actionable business recommendations

---

## Dataset

The dataset contains customer reviews collected from ShopEase Europe.

### Features

- Customer Review
- Sentiment
- Product Category
- Country
- Rating
- Timestamp
- Review Id

### Target Variable
## Sentiment 
- Negative
- Neutral
- Positive

---

## Project Workflow

```
Data Collection
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis (EDA)
        │
        ▼
Text Preprocessing
        │
        ▼
Feature Engineering (TF-IDF)
        │
        ▼
Baseline Models
• Naïve Bayes
• Logistic Regression
• Linear SVM
        │
        ▼
DistilBERT Fine-Tuning
        │
        ▼
Model Evaluation
        │
        ▼
Business Insights
        │
        ▼
Power BI Dashboard
        │
        ▼
Streamlit Deployment
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Hugging Face Transformers
- PyTorch
- Streamlit
- Plotly
- Matplotlib
- Power BI

---

## Machine Learning Models

The following models were trained and evaluated:

- Naïve Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)
- DistilBERT (Transformer)

---

## Model Performance

| Model | Accuracy | ROC-AUC |
|--------|----------|----------|
| Naïve Bayes | 83.29% | 94.36% |
| Logistic Regression | 84.95% | 94.51% |
| Linear SVM | 85.62% | 93.99% |
| **DistilBERT** | **90.93%** | **95.72%** |

### Final Model

✅ **DistilBERT**

DistilBERT achieved the highest overall accuracy and delivered the best sentiment classification performance.

## Model

The fine-tuned DistilBERT model is hosted on Hugging Face Hub.

Model Repository:

https://huggingface.co/YourUsername/Customer-Sentiment-DistilBERT

---

## Key Findings

- Approximately **69%** of customer reviews were Negative.
- Most customers give the lowest possible rating.
- Almost all feedback is written in English and is mostly negative.
- Customer Service was the most frequent complaint.
- Delivery issues were a major source of dissatisfaction.
- Fast delivery, competitive pricing, and good customer service were the main drivers of positive sentiment.
- DistilBERT outperformed all baseline models.
- Neutral reviews remained difficult to classify because they represented only a small proportion of the dataset.

---

## Business Recommendations

- Improve customer service response times.
- Reduce delivery delays.
- Simplify refund and return processes.
- Replicate practices from high-performing products and services.
- Monitor customer sentiment continuously using the deployed application.
- Learn from Positive Feedback.

---

## Results

This project demonstrates that transformer-based models significantly outperform traditional machine learning models for customer sentiment analysis.

The final DistilBERT model achieved:

- **91.33% Accuracy**
- **95.78% ROC-AUC**

making it the recommended model for deployment.

---

## Dashboard

The Power BI dashboard provides interactive insights into:

- Excutive summary
- Sentiment analysis
- Category analysis
- Geographical analysis
- Sentiment drivers


---

## Streamlit Application

The Streamlit application enables users to:

- Upload customer reviews
- Predict sentiment in real time
- Visualise prediction results
- Analyse customer feedback interactively

---

## Author

**Olayemi Balogun**

Data Analytics | Data Science | Machine Learning | Natural Language Processing

LinkedIn: *(www.linkedin.com/in/olayemi-balogun31)*

---

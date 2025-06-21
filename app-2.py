import streamlit as st
import joblib
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Load model
model = joblib.load("random_forest_model.pkl")

# Clean text
def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

# Sentiment score
def get_score(text):
    cleaned = clean_text(text)
    score = sid.polarity_scores(cleaned)['compound']
    return score

# Sentiment prediction
def predict_sentiment(text):
    score = get_score(text)
    prediction = model.predict([[score]])[0]
    return prediction

# Streamlit UI
st.title("📊 Analisis Sentimen Media Sosial untuk Prediksi Tren Pasar")
text_input = st.text_area("Masukkan Judul Postingan Reddit / Kalimat Analisis")

if st.button("Prediksi Sentimen"):
    if text_input:
        sentiment = predict_sentiment(text_input)
        st.success(f"Prediksi Sentimen: **{sentiment.upper()}**")
    else:
        st.warning("Tolong masukkan teks terlebih dahulu.")

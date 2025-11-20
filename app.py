import streamlit as st
import joblib
import re
import numpy as np
import os

# Load models and vectorizers from model/ folder
lda = joblib.load(os.path.join('model', 'lda_topic_model.joblib'))
vectorizer = joblib.load(os.path.join('model', 'topic_vectorizer.joblib'))
best_model = joblib.load(os.path.join('model', 'sentiment_best_model.joblib'))
tfidf = joblib.load(os.path.join('model', 'sentiment_tfidf_vectorizer.joblib'))

# Example topic labels
topic_labels = {
    0: "Positive Course Experience and Engagement",
    1: "Suggestions for Improvement and Challenges",
    2: "Teaching Methods and Learning Activities",
    3: "Course Content and Structure",
    4: "Feedback and Assessment"
}

# Preprocessing function (same as used for training)
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Optionally remove stopwords if needed (can load stopwords as in notebook)
    return text

# Prediction function
def predict_topic_sentiment(text):
    clean = preprocess(text)
    vec_topic = vectorizer.transform([clean])
    topic_idx = lda.transform(vec_topic).argmax(axis=1)[0]
    topic_label = topic_labels.get(topic_idx, f"Topic {topic_idx}")
    topic_words = ', '.join([vectorizer.get_feature_names_out()[i] for i in lda.components_[topic_idx].argsort()[:-6:-1]])
    vec_sentiment = tfidf.transform([clean])
    sentiment = best_model.predict(vec_sentiment)[0]
    return topic_label, topic_words, sentiment

# Streamlit UI
st.title('Course Evaluation Topic & Sentiment Analyzer')
st.write('Enter a student evaluation below to predict its main topic and sentiment:')
user_input = st.text_area('Student Evaluation')
if st.button('Analyze') and user_input:
    topic_label, topic_words, sentiment = predict_topic_sentiment(user_input)
    st.subheader('Predicted Topic')
    st.write(topic_label)
    st.write('Top Words:', topic_words)
    st.subheader('Predicted Sentiment')
    st.write(sentiment.capitalize())
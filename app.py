# streamlit_app.py
import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# 1. Load saved resources
@st.cache(allow_output_mutation=True)
def load_resources():
    # Tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    # Model
    model = tf.keras.models.load_model('sentiment_model')
    # Summarizer
    summarizer = pipeline(
        'summarization', model='t5-small', tokenizer='t5-small', framework='tf'
    )
    return tokenizer, model, summarizer

tokenizer, model, summarizer = load_resources()
vocab_size = tokenizer.num_words or 10000
max_length = model.input_shape[1]

# 2. Streamlit UI
st.title("🎥 IMDB Sentiment Analysis Demo")
review = st.text_area("Enter your movie review:", height=200)
summarize = st.checkbox("Summarize review before prediction", value=True)

if st.button("Analyze"):
    if not review.strip():
        st.error("Please provide a review.")
    else:
        # Optional summarization
        if summarize:
            summary = summarizer(review, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        else:
            summary = review

        # Tokenize & pad
        seq = tokenizer.texts_to_sequences([summary])
        padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
        prob = float(model.predict(padded)[0][0])
        label = 'Positive' if prob >= 0.5 else 'Negative'

        # Display
        st.subheader("📝 Summary")
        st.write(summary)
        st.subheader("🖥️ Prediction")
        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence:** {prob:.2f}")

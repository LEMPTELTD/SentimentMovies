# streamlit_app.py
import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Summarization requires transformers; disable if not installed
try:
    from transformers import pipeline
    HAS_SUMMARY = True
except ImportError:
    HAS_SUMMARY = False

@st.cache(allow_output_mutation=True)
def load_resources():
    # Tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    # Model
    model = tf.keras.models.load_model('sentiment_model')
    # Summarizer if available
    summarizer = None
    if HAS_SUMMARY:
        summarizer = pipeline('summarization', model='t5-small', tokenizer='t5-small', framework='tf')
    return tokenizer, model, summarizer

tokenizer, model, summarizer = load_resources()
vocab_size = getattr(tokenizer, 'num_words', 10000)
max_length = model.input_shape[1]

st.title("IMDB Sentiment Analysis")
st.write("Summarization is " + ("enabled" if HAS_SUMMARY else "disabled") + ".")
review = st.text_area("Enter review:", height=200)
summarize = st.checkbox("Summarize", value=HAS_SUMMARY, disabled=not HAS_SUMMARY)

if st.button("Analyze"):
    if not review.strip():
        st.error("Please provide review text.")
    else:
        if summarize and summarizer:
            summary = summarizer(review, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        else:
            summary = review
        seq = tokenizer.texts_to_sequences([summary])
        padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
        prob = float(model.predict(padded)[0][0])
        label = "Positive" if prob >= 0.5 else "Negative"
        st.subheader("Summary")
        st.write(summary)
        st.subheader("Prediction")
        st.write(f"{label} (Confidence: {prob:.2f})")

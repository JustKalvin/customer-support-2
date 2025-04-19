import streamlit as st
import random
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from langdetect import detect_langs

# Download NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load model dan tokenizer
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./NLP_FINAL_MODEL.h5")

@st.cache_resource
def load_tokenizer():
    with open("./tokenizer.pkl", "rb") as handle:
        return pickle.load(handle)

@st.cache_resource
def load_tfidf():
    data = pd.read_csv('./Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data["instruction"])
    return vectorizer, data

# Load semua komponen
model = load_model()
tokenizer = load_tokenizer()
vectorizer, data = load_tfidf()
translator = Translator()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

encoded_intent = [
    "cancel_order", "change_order", "change_shipping_address", "check_cancellation_fee",
    "check_invoice", "check_payment_methods", "check_refund_policy", "complaint",
    "contact_customer_service", "contact_human_agent", "create_account", "delete_account",
    "delivery_options", "delivery_period", "edit_account", "get_invoice", "get_refund",
    "newsletter_subscription", "payment_issue", "place_order", "recover_password",
    "registration_problems", "review", "set_up_shipping_address", "switch_account",
    "track_order", "track_refund"
]

# Preprocessing function
def preprocessing(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# TF-IDF response retrieval
def get_best_response(user_input, predicted_intent):
    filtered_data = data[data['intent'] == predicted_intent]
    if filtered_data.empty:
        return "I'm sorry, I don't understand your request."

    tfidf_matrix = vectorizer.transform(filtered_data["instruction"])
    user_input_tfidf = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)
    best_idx = similarities.argmax()

    return filtered_data.iloc[best_idx]["response"]

# Prediksi chatbot
def predict_text(text):
    try:
        lang_detect = detect_langs(text)
        lang_code = lang_detect[0].lang
        if lang_detect[0].prob < 0.9:
            lang_code = 'en'

        if lang_code != 'en':
            text = translator.translate(text, src=lang_code, dest="en").text

        preprocessed_text = preprocessing(text)
        sequence = tokenizer.texts_to_sequences([preprocessed_text])

        if not sequence or not sequence[0]:  
            return "unknown_intent", "I'm a bit confused, please use another sentence."

        x = pad_sequences(sequence)
        prediction = model.predict(x)
        predicted_intent = encoded_intent[np.argmax(prediction[0])]
        response = get_best_response(text, predicted_intent)

        response_translated = translator.translate(response, src='en', dest=lang_code).text  

        return predicted_intent, response_translated

    except Exception as e:
        return "unknown_intent", "Sorry, I can't understand what you typed."

# ------------------------- Streamlit UI -------------------------

# Styling
st.markdown("""
    <style>
    .chat-container {
        max-width: 700px;
        margin: auto;
    }
    .chat-user {
        text-align: right;
        color: white;
        background-color: #0078FF;
        padding: 8px;
        border-radius: 10px;
        display: inline-block;
        max-width: 80%;
    }
    .chat-bot {
        text-align: left;
        color: black;
        background-color: #F0F0F0;
        padding: 8px;
        border-radius: 10px;
        display: inline-block;
        max-width: 80%;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🤖 AI Chatbot with LSTM & TF-IDF</h1>", unsafe_allow_html=True)

# Inisialisasi chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Menampilkan chat history
st.write("<div class='chat-container'>", unsafe_allow_html=True)
for message in st.session_state.messages:
    role_class = "chat-user" if message["role"] == "user" else "chat-bot"
    st.markdown(f"<p class='{role_class}'>{message['content']}</p>", unsafe_allow_html=True)
st.write("</div>", unsafe_allow_html=True)

# Input pengguna
if user_input := st.chat_input("Ask me anything..."):
    # Menampilkan pesan pengguna
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"<p class='chat-user'>{user_input}</p>", unsafe_allow_html=True)

    # Prediksi chatbot
    intent, response = predict_text(user_input)

    # Menampilkan respons chatbot
    st.session_state.messages.append({"role": "bot", "content": response})
    st.markdown(f"<p class='chat-bot'>Intent : {intent} <br> {response}</p>", unsafe_allow_html=True)

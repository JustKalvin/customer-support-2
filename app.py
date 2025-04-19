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
import requests
from io import BytesIO
import base64
import json
from streamlit_webrtc import webrtc_streamer, ClientSettings, AudioProcessor
import soundfile as sf
import os

# Download NLTK dependencies (pastikan ini hanya dijalankan sekali)
try:
    nltk.data.find('corpora/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
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

from pathlib import Path

# Load background image as base64
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Panggil fungsi ini dengan nama file gambarnya
set_bg_from_local("NLP_Streamlit_Background.jpeg")

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

# ------------------------- Streamlit UI with Speech Recognition -------------------------
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

st.markdown("""
<style>
/* Ubah font seluruh aplikasi */
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    font-size: 16px;
}

.stApp {
    background-color: rgba(255,255,255,0);
}

.chat-container {
    max-width: 700px;
    margin: auto;
    padding: 20px;
}

/* Efek fade-in untuk bubble */
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

.chat-user, .chat-bot {
    padding: 12px 18px;
    border-radius: 20px;
    margin: 10px 0;
    display: inline-block;
    max-width: 80%;
    animation: fadeIn 0.5s ease-in-out;
}

/* Chat bubble user */
.chat-user {
    text-align: right;
    background-color: #0078FF;
    color: white;
    margin-left: auto;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
}

/* Chat bubble bot */
.chat-bot {
    text-align: left;
    background-color: #f9f9f9;
    color: #333;
    margin-right: auto;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.15);
}

/* Input field styling */
input[type="text"] {
    background-color: #1a1a1a;
    padding: 10px;
    border: 2px solid #ffffff;
    border-radius: 6px;
    color: #ffffff;
}

/* Tombol rekam lebih modern */
button[kind="primary"] {
    background: linear-gradient(90deg, #0078FF 0%, #00C6FF 100%);
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
    transition: 0.3s;
}
button[kind="primary"]:hover {
    transform: scale(1.05);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🤖 AI Customer Support</h1>", unsafe_allow_html=True)

# State untuk menyimpan teks yang dikenali dari suara
if "recognized_text" not in st.session_state:
    st.session_state["recognized_text"] = ""
if "audio_data" not in st.session_state:
    st.session_state["audio_data"] = None

# Konfigurasi WebRTC
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Kelas untuk memproses audio dan menyimpan ke state
class AudioToTextHandler(AudioProcessor):
    def __init__(self):
        self.audio_buffer = []
        self.sample_rate = 16000  # Contoh sample rate
        self.recording = False

    def recv(self, frame):
        if self.recording:
            sound_wave = frame.to_ndarray(format="s16").reshape(-1)
            self.audio_buffer.extend(sound_wave.tobytes())
        return frame.to_ndarray()

    def on_ended(self):
        if self.audio_buffer:
            st.session_state["audio_data"] = b"".join(self.audio_buffer)
            st.experimental_rerun()

# Fungsi untuk memulai perekaman
def start_recording():
    st.session_state["audio_data"] = None
    st.session_state["recognized_text"] = ""
    st.session_state["audio_handler"].audio_buffer = []
    st.session_state["audio_handler"].recording = True

# Fungsi untuk menghentikan perekaman
def stop_recording():
    st.session_state["audio_handler"].recording = False

# Inisialisasi audio handler di state
if "audio_handler" not in st.session_state:
    st.session_state["audio_handler"] = AudioToTextHandler()

webrtc_streamer(
    key="speech_to_text",
    client_settings=WEBRTC_CLIENT_SETTINGS,
    audio_processor_factory=lambda: st.session_state["audio_handler"],
    media_stream_constraints={"audio": True, "video": False},
    on_ended=st.session_state["audio_handler"].on_ended,
)

col1, col2 = st.columns(2)
if col1.button("Mulai Rekam", on_click=start_recording):
    st.info("Mulai berbicara...")

if col2.button("Hentikan Rekam", on_click=stop_recording):
    st.success("Perekaman selesai. Anda bisa mengirim pesan sekarang.")

# Input area (menggunakan nilai dari state jika ada audio yang direkam)
user_input = st.text_input("Ask me anything (or speak)...", key="chat_input", value=st.session_state.get("recognized_text", ""))

# Proses input pengguna (baik diketik atau dari ucapan)
if st.button("Kirim"):
    if user_input:
        # Menampilkan pesan pengguna
        st.session_state.messages = [{"role": "user", "content": user_input}] if "messages" not in st.session_state else st.session_state.messages + [{"role": "user", "content": user_input}]
        st.markdown(f"<p class='chat-user'>{user_input}</p>", unsafe_allow_html=True)

        # Prediksi chatbot
        intent, response_translated = predict_text(user_input)

        # Menampilkan respons chatbot
        st.session_state.messages = st.session_state.messages + [{"role": "bot", "content": response_translated}]
        st.markdown(f"<p class='chat-bot'>Intent : {intent} <br> {response_translated}</p>", unsafe_allow_html=True)
        st.session_state["recognized_text"] = "" # Bersihkan input setelah mengirim
    else:
        st.warning("Mohon masukkan teks atau rekam suara terlebih dahulu!")

# Menampilkan chat history
st.write("<div class='chat-container'>", unsafe_allow_html=True)
if "messages" in st.session_state:
    for message in st.session_state.messages:
        role_class = "chat-user" if message["role"] == "user" else "chat-bot"
        st.markdown(f"<p class='{role_class}'>{message['content']}</p>", unsafe_allow_html=True)
st.write("</div>", unsafe_allow_html=True)

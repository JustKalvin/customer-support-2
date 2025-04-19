import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.metrics import Recall, Precision, F1Score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator, LANGUAGES
from langdetect import detect, detect_langs
import pickle
import streamlit as st
import base64
import speech_recognition as sr
from streamlit_audio_recorder import audio_recorder


def get_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Ganti path ke gambar kamu
image_base64 = get_base64("purple-mountain-landscape.jpg")

# Masukin ke CSS
page_bg_color = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{image_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""

st.markdown(page_bg_color, unsafe_allow_html=True)

# Load model dan tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("NLP_FINAL_MODEL.h5")  # Ganti dengan path model Anda
    return model

@st.cache_resource
def load_tokenizer():
  with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)  # Load tokenizer dari file
  return tokenizer  # Ganti dengan tokenizer yang sudah di-train

@st.cache_resource
def load_tfidf():
    data = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(data["instruction"])
    return vectorizer, data

# Load semua komponen
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
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
    print(f'predicted_intent : {predicted_intent}')
    vectorizer = TfidfVectorizer()
    theData = data[data['intent'] == predicted_intent]
    tfidf_matrix = vectorizer.fit_transform(data["instruction"])
    
    user_input_tfidf = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_input_tfidf, tfidf_matrix)

    best_idx = similarities.argmax()

    return data.iloc[best_idx]["response"]

translator = Translator()  

def predict_text(text):
    try:
     
        lang_detect = detect_langs(text)
        lang_code = lang_detect[0].lang 
        print(f'prob lang_detect : {lang_detect[0].prob}')

        if lang_detect[0].prob < 0.9:
            lang_code = 'en' 

        print(f'lang_detect : {lang_code}')


        if lang_code != 'en':
            text = translator.translate(text, src=lang_code, dest="en").text

        preprocessed_text = preprocessing(text)
        print("Preprocessed text:", preprocessed_text)  

        sequence = tokenizer.texts_to_sequences([preprocessed_text])  
        print("Tokenized sequence:", sequence)  

        if not sequence or not sequence[0]:  
            raise ValueError("I'm a bit confused, please use another sentence.")  

        x = pad_sequences(sequence)
        print("Padded sequence shape:", x.shape)  

        prediction = model.predict(x)
        response = get_best_response(text, encoded_intent[np.argmax(prediction[0])][0])

        response_translated = translator.translate(response, src='en', dest=lang_code).text  

        return encoded_intent[np.argmax(prediction[0])], response_translated
    
    except Exception as e:
        print(f"Error: {e}")
        return "unknown_intent", "Sorry, I can't understand what you typing. Please type another sentence."


# text = 'saya ingin membuat akun, bagaimana cara membuatnya?'
# intent, response = predict_text(text)
# print(f'Intent : {intent}\nResponse : {response}')

# Streamlit UI
st.markdown(
    """
    <style>
    h1 {
        font-size: 48px; /* Atur ulang ukuran teks */
        font-weight: bold;
        color: white;
        text-shadow: 1px 1px 2px black; /* Bayangan lebih halus */
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🧠 Multi-Language Chatbot with LSTM & TF-IDF</h1>", unsafe_allow_html=True)
st.markdown('<h1>📝 Input Your Question Below : </h1>', unsafe_allow_html = True)

# Voice input section
st.markdown('<h1>🎤 Or Use Voice Input : </h1>', unsafe_allow_html = True)
audio_bytes = audio_recorder(text="Tekan untuk Rekam", icon_size="2x")

def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    try:
        with audio_file as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        return f"Speech recognition error: {e}"

if audio_bytes:
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_bytes)
    audio_file = sr.AudioFile("temp_audio.wav")
    user_input = speech_to_text(audio_file)
    st.success(f"🗣️ Recognized Text: {user_input}")

# Manual text input fallback
user_input = st.text_input('Input Your Question : ', value=user_input if audio_bytes else "")

if st.button("Kirim"):
    if user_input:
        intent, response_translated = predict_text(user_input)

        st.markdown(
            f"""
            <div style="border: 2px solid white; padding: 10px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.1); text-align: center;">
                <p style="color: white; font-size: 22px; font-weight: bold; text-shadow: 2px 2px 4px black;">
                    🎯 Intent: {intent}
                </p>
                <p style="color: white; font-size: 20px; font-weight: bold; text-shadow: 2px 2px 4px black;">
                    💬 Response: {response_translated}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Mohon masukkan teks terlebih dahulu!")




user_input = st.text_input('Input Your Question : ')
if st.button("Kirim"):  
    if user_input:
        intent, response_translated = predict_text(user_input)

        st.markdown(
            f"""
            <div style="border: 2px solid white; padding: 10px; border-radius: 10px; background-color: rgba(255, 255, 255, 0.1); text-align: center;">
                <p style="color: white; font-size: 22px; font-weight: bold; text-shadow: 2px 2px 4px black;">
                    🎯 Intent: {intent}
                </p>
                <p style="color: white; font-size: 20px; font-weight: bold; text-shadow: 2px 2px 4px black;">
                    💬 Response: {response_translated}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Mohon masukkan teks terlebih dahulu!")


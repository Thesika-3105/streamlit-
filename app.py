# app.py
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import plotly.express as px
import speech_recognition as sr
from gtts import gTTS
import os
from io import BytesIO
from pydub import AudioSegment

st.set_page_config(page_title="AI Semantic Job Search & Scheme Recommender", layout="wide")

# ------------------------------
# Load Data & Models
# ------------------------------
@st.cache_data
def load_data():
    nco_df = pickle.load(open("nco_df.pkl", "rb"))
    scheme_df = pickle.load(open("scheme_df.pkl", "rb"))
    scheme_embeddings = np.load("scheme_embeddings.npy")
    nco_embeddings = np.load("nco_embeddings.npy")
    index = faiss.read_index("nco_faiss.index")
    return nco_df, scheme_df, scheme_embeddings, nco_embeddings, index

nco_df, scheme_df, scheme_embeddings, nco_embeddings, index = load_data()

model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------
# Helper Functions
# ------------------------------
def semantic_search(query, top_k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype('float32'), top_k)
    results = nco_df.iloc[I[0]].copy()
    return results

def get_schemes(nco_code):
    schemes = scheme_df[scheme_df['NCO_Code'] == nco_code]
    return schemes

def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return text
        except:
            st.warning("Could not recognize speech. Please try again.")
            return ""

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = AudioSegment.from_file(fp, format="mp3")
    return audio

# ------------------------------
# UI
# ------------------------------
st.title("🔍 AI Semantic Job Search & Government Scheme Recommender")
st.markdown("Search for jobs and get recommended government schemes easily!")

# Input Mode
input_mode = st.radio("Input Mode", ["Text", "Voice"])

if input_mode == "Text":
    query = st.text_input("Enter job role or skill:")
else:
    if st.button("Start Voice Input"):
        query = recognize_speech()
        st.success(f"You said: {query}")
    else:
        query = ""

# Language Selection
lang = st.selectbox("Select Output Language", ["en", "ta", "hi", "ml", "te"])

# ------------------------------
# Process Query
# ------------------------------
if query:
    st.subheader("🔹 Semantic Job Search Results")
    results = semantic_search(query)
    
    # Translate job titles
    results_display = results.copy()
    results_display['Job_Title'] = results_display['Job_Title'].apply(lambda x: translate_text(x, lang))
    st.dataframe(results_display[['Job_Title', 'NCO_Code', 'Job_Description']])
    
    # Schemes for top result
    top_nco = results.iloc[0]['NCO_Code']
    schemes = get_schemes(top_nco)
    schemes_display = schemes.copy()
    schemes_display['Scheme_Name'] = schemes_display['Scheme_Name'].apply(lambda x: translate_text(x, lang))
    
    st.subheader("🏛 Recommended Government Schemes")
    st.dataframe(schemes_display[['Scheme_Name', 'Scheme_Objective', 'Eligibility']])
    
    # Charts
    st.subheader("📊 Interactive Analytics")
    # Top 10 job titles by frequency
    top_jobs = nco_df['Job_Title'].value_counts().nlargest(10).reset_index()
    top_jobs.columns = ['Job_Title', 'Count']
    fig = px.bar(top_jobs, x='Job_Title', y='Count', title="Top 10 Job Titles")
    st.plotly_chart(fig)
    
    # Optionally, play TTS of top job
    if st.button("🔊 Listen to Top Job Title"):
        audio = text_to_speech(results.iloc[0]['Job_Title'])
        audio.export("temp.wav", format="wav")
        st.audio("temp.wav")
        os.remove("temp.wav")

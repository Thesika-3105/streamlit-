# app.py
# -------------------------------
# AI-enabled Semantic NCO Search + Govt Scheme Recommender
# Files expected (already with you):
#   - nco_df.pkl (preferred) or nco_cleaned.csv
#   - scheme_df.pkl (preferred) or govt_schemes.csv
#   - nco_faiss.index
#   - nco_embeddings.npy
#   - scheme_embeddings.npy   (kept for future; not strictly needed here)
#
# Optional: You may also include a logo at ./logo.png to show in the header.

import os
import io
import time
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from typing import List, Dict, Any

# ML + Search
import faiss
from sentence_transformers import SentenceTransformer

# Translation + Voice
from googletrans import Translator
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment

# -------------------------------
# Page Config & Theming
# -------------------------------
st.set_page_config(
    page_title="NCO Semantic Search + Scheme Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Subtle CSS polish
st.markdown("""
<style>
:root {
  --radius: 16px;
}
.block-container {padding-top: 1.8rem; padding-bottom: 2rem;}
/* Card look */
.card {
  border: 1px solid rgba(49,51,63,0.2);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  background: rgba(250, 250, 252, 0.8);
  box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.card h4 { margin: 0 0 .25rem 0; }
.badge {
  display:inline-block; padding:.2rem .6rem; border-radius: 999px;
  border:1px solid rgba(49,51,63,.2); font-size:.8rem; opacity:.9;
}
.relevance {
  font-size:.85rem; opacity:.75;
}
hr.soft { border:none; height:1px; background:linear-gradient(90deg,transparent,rgba(0,0,0,.12),transparent); margin:.75rem 0; }
.small { font-size:.9rem; opacity:.9; }
.kpi {
  border-radius: var(--radius);
  padding: .8rem 1rem;
  background: rgba(242, 245, 255, .7);
  border: 1px dashed rgba(49,51,63,.25);
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Config
# -------------------------------
DATA_FILES = {
    "nco_df_pkl": "nco_df.pkl",
    "nco_csv": "nco_cleaned.csv",
    "scheme_df_pkl": "scheme_df.pkl",
    "scheme_csv": "govt_schemes.csv",
    "nco_index": "nco_faiss.index",
    "nco_embeddings": "nco_embeddings.npy",
    "scheme_embeddings": "scheme_embeddings.npy"
}

LANG_MAP = {
    "English": "en",
    "Tamil (தமிழ்)": "ta",
    "Hindi (हिन्दी)": "hi",
    "Malayalam (മലയാളം)": "ml",
    "Telugu (తెలుగు)": "te"
}

# -------------------------------
# Loaders (cached)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_faiss_index(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index file not found: {path}")
    return faiss.read_index(path)

@st.cache_data(show_spinner=False)
def load_dataframe(pkl_path: str, csv_path: str, expected_cols: List[str] = None) -> pd.DataFrame:
    df = None
    if os.path.exists(pkl_path):
        df = pd.read_pickle(pkl_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Neither {pkl_path} nor {csv_path} was found.")
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            st.warning(f"Missing expected columns {missing} in {pkl_path or csv_path}. App may still work if your schema differs.")
    return df

@st.cache_resource(show_spinner=False)
def load_embeddings(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    return np.load(path)

# -------------------------------
# Helpers
# -------------------------------
@st.cache_resource(show_spinner=False)
def get_translator():
    return Translator()

def safe_translate(text: str, dest_lang: str, translator: Translator) -> str:
    if not text or dest_lang == "en":
        return text
    try:
        res = translator.translate(text, dest=dest_lang)
        return res.text
    except Exception:
        # fallback: return original if translate fails
        return text

def translate_df(df: pd.DataFrame, cols: List[str], dest_lang: str, translator: Translator) -> pd.DataFrame:
    if dest_lang == "en":
        return df
    df2 = df.copy()
    for c in cols:
        if c in df2.columns:
            df2[c] = df2[c].astype(str).apply(lambda t: safe_translate(t, dest_lang, translator))
    return df2

def l2_to_relevance(l2_dist: float) -> float:
    # Convert L2 distance to an intuitive "relevance" in [0..100]
    # Smaller distance => higher relevance. This mapping is heuristic.
    rel = max(0.0, 100.0 - float(l2_dist) * 10.0)
    return round(rel, 1)

def search_nco(query: str, model: SentenceTransformer, index, nco_df: pd.DataFrame, top_k: int = 5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb).astype("float32"), top_k)
    hits = nco_df.iloc[I[0]].copy()
    hits["distance"] = D[0]
    hits["relevance"] = hits["distance"].apply(l2_to_relevance)
    return hits

def get_scheme_matches_for_nco(nco_code: Any, scheme_df: pd.DataFrame) -> pd.DataFrame:
    # Expect a column that links schemes to NCO, e.g., "NCO Code" or "nco_code"
    link_cols = [c for c in scheme_df.columns if c.lower().replace(" ", "") in ("ncocode","nco_code","ncoid")]
    if not link_cols:
        # If no explicit link column, try best-effort: exact string match in a generic "NCO Code"
        link_cols = ["NCO Code"] if "NCO Code" in scheme_df.columns else []
    if not link_cols:
        # No linking column—return all schemes (or empty)
        return scheme_df.head(0)
    col = link_cols[0]
    return scheme_df[scheme_df[col].astype(str) == str(nco_code)]

def read_speech_from_file(uploaded_file) -> str:
    # Convert to wav for SpeechRecognition via pydub if needed
    # supported: wav, mp3, m4a, webm, etc. (pydub ffmpeg support required)
    audio_bytes = uploaded_file.read()
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
    return recognizer.recognize_google(audio_data)

def synthesize_tts(text: str, lang_code: str) -> bytes:
    tts = gTTS(text=text, lang=lang_code)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()

# -------------------------------
# Load data/models (one-time)
# -------------------------------
with st.spinner("Loading data and models..."):
    model = load_model()
    nco_index = load_faiss_index(DATA_FILES["nco_index"])
    nco_df = load_dataframe(DATA_FILES["nco_df_pkl"], DATA_FILES["nco_csv"])
    scheme_df = load_dataframe(DATA_FILES["scheme_df_pkl"], DATA_FILES["scheme_csv"])
    # not strictly used, but preloaded for future features/analysis
    if os.path.exists(DATA_FILES["nco_embeddings"]):
        _nco_embeddings = load_embeddings(DATA_FILES["nco_embeddings"])
    if os.path.exists(DATA_FILES["scheme_embeddings"]):
        _scheme_embeddings = load_embeddings(DATA_FILES["scheme_embeddings"])
    translator = get_translator()

# Try to infer core columns
NCO_CODE_COL = next((c for c in nco_df.columns if c.lower().replace(" ", "") in ("ncocode","nco_code","code")), None)
TITLE_COL = next((c for c in nco_df.columns if c.lower() in ("job title","job_title","title","occupation","role")), None)
DESC_COL = next((c for c in nco_df.columns if "desc" in c.lower()), None)

# -------------------------------
# Sidebar Controls
# -------------------------------
with st.sidebar:
    st.image("logo.png", width=120, caption="") if os.path.exists("logo.png") else None
    st.markdown("### ⚙️ Settings")
    lang_label = st.selectbox("Output language", list(LANG_MAP.keys()), index=0)
    dest_lang = LANG_MAP[lang_label]

    top_k = st.slider("Number of job matches (Top-K)", min_value=3, max_value=15, value=7, step=1)

    st.markdown("### 🎤 Voice Search")
    st.caption("Upload a short voice note (wav/mp3/m4a/webm). We'll transcribe it.")
    audio_file = st.file_uploader("Upload audio", type=["wav","mp3","m4a","webm"], label_visibility="collapsed")

    enable_tts = st.checkbox("🔊 Read results aloud (TTS)")
    st.markdown("---")
    st.markdown("### 📈 Analytics")
    show_analytics = st.checkbox("Show scheme & job analytics", value=True)

# -------------------------------
# Header
# -------------------------------
colA, colB = st.columns([0.7, 0.3])
with colA:
    st.title("🧭 NCO Semantic Search & Scheme Recommender")
    st.write("Type your query (e.g., *\"data entry job\"*, *\"electrician fresher\"*) or upload a short voice note. Get the closest NCO roles and matching government schemes.")
with colB:
    st.markdown('<div class="kpi"><b>Ready:</b> Indexed jobs & schemes loaded.<br><b>Search Method:</b> Semantic (Sentence Transformers + FAISS)</div>', unsafe_allow_html=True)

# -------------------------------
# Input Row
# -------------------------------
st.markdown("### 🔍 Search")
default_q = "computer operator"
query = st.text_input("Describe your target job, skills, sector, or duties:", value=default_q, placeholder="e.g., 'data entry', 'electrical technician', 'civil site supervisor'")
voice_transcript = None

if audio_file:
    with st.spinner("Transcribing audio…"):
        try:
            voice_transcript = read_speech_from_file(audio_file)
            st.success(f"Transcribed: “{voice_transcript}”")
            if (not query) or (query == default_q):
                query = voice_transcript
        except Exception as e:
            st.error(f"Could not transcribe audio: {e}")

go_search = st.button("Search", type="primary") or (st.session_state.get("auto_search", False))

# -------------------------------
# Search & Results
# -------------------------------
if go_search and query.strip():
    with st.spinner("Finding the best matching NCO roles…"):
        hits = search_nco(query.strip(), model, nco_index, nco_df, top_k=top_k)

    # Translate job results if needed
    trans_cols = []
    if TITLE_COL: trans_cols.append(TITLE_COL)
    if DESC_COL: trans_cols.append(DESC_COL)
    hits_display = translate_df(hits, trans_cols, dest_lang, translator) if dest_lang else hits

    # Results UI
    st.markdown("### ✅ Matches")
    st.caption(f"Language: **{lang_label}** · Results: **Top {top_k}**")
    for i, row in hits_display.reset_index(drop=True).iterrows():
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)

            title_txt = str(row.get(TITLE_COL, f"Job #{i+1}")).strip()
            desc_txt = str(row.get(DESC_COL, "")).strip()
            code_txt = str(row.get(NCO_CODE_COL, "—"))

            c1, c2, c3 = st.columns([0.60, 0.20, 0.20])
            with c1:
                st.markdown(f"#### {title_txt}")
                st.markdown(f"<span class='badge'>NCO Code: {code_txt}</span> &nbsp; <span class='relevance'>Relevance: {row.get('relevance', '—')}%</span>", unsafe_allow_html=True)
            with c2:
                add_to_tts = st.checkbox(f"Read this", key=f"tts_{i}") if enable_tts else None
            with c3:
                st.write("")

            if desc_txt:
                st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
                st.markdown(f"**Description:** {desc_txt}")

            # Schemes
            with st.expander("🏛 View matching government schemes"):
                matches = get_scheme_matches_for_nco(row.get(NCO_CODE_COL, ""), scheme_df)
                if len(matches) == 0:
                    st.info("No schemes were explicitly linked to this NCO code in your dataset.")
                else:
                    # Identify useful columns to show
                    preferred_cols = []
                    for guess in ["Scheme Name","Scheme","Title","Name","Eligibility","Benefit","Link","NCO Code"]:
                        if guess in matches.columns and guess not in preferred_cols:
                            preferred_cols.append(guess)
                    # Guarantee we show something
                    if not preferred_cols:
                        preferred_cols = list(matches.columns[:6])

                    # Translate scheme fields
                    matches_display = translate_df(matches, preferred_cols, dest_lang, translator)

                    st.dataframe(matches_display[preferred_cols], use_container_width=True, hide_index=True)

            st.markdown('</div>', unsafe_allow_html=True)

    # Optional TTS for all checked items
    if enable_tts:
        to_speak = []
        for i, row in hits_display.reset_index(drop=True).iterrows():
            if st.session_state.get(f"tts_{i}", False):
                title_txt = str(row.get(TITLE_COL, f"Job #{i+1}")).strip()
                code_txt = str(row.get(NCO_CODE_COL, "—"))
                to_speak.append(f"{i+1}. {title_txt}. N C O code {code_txt}.")
        if to_speak:
            speak_text = "  ".join(to_speak)
            with st.spinner("Synthesizing speech…"):
                try:
                    audio_bytes = synthesize_tts(speak_text, LANG_MAP.get(lang_label, "en"))
                    st.audio(audio_bytes, format="audio/mp3", start_time=0)
                    st.download_button("Download audio (MP3)", data=audio_bytes, file_name="nco_results.mp3", mime="audio/mpeg")
                except Exception as e:
                    st.error(f"TTS failed: {e}")

# -------------------------------
# Analytics
# -------------------------------
if show_analytics:
    st.markdown("### 📊 Insights & Coverage")
    col1, col2 = st.columns(2)

    with col1:
        # Top NCO titles by frequency (if duplicates exist) or by simple sample
        title_col = TITLE_COL or next((c for c in nco_df.columns if "title" in c.lower() or "occupation" in c.lower()), None)
        if title_col:
            vc = nco_df[title_col].value_counts().head(10).reset_index()
            vc.columns = ["Job Title","Count"]
            fig = px.bar(vc, x="Job Title", y="Count", title="Most Frequent NCO Titles (Top 10)")
            fig.update_layout(xaxis_title="", yaxis_title="Count", bargap=0.2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Could not infer the title column for NCO data to build this chart.")

    with col2:
        # Schemes linked per NCO code
        link_col = next((c for c in scheme_df.columns if c.lower().replace(" ", "") in ("ncocode","nco_code","ncoid")), None)
        if link_col:
            vc2 = scheme_df[link_col].value_counts().head(10).reset_index()
            vc2.columns = ["NCO Code","# Schemes"]
            fig2 = px.bar(vc2, x="NCO Code", y="# Schemes", title="Top NCO Codes by Scheme Coverage")
            fig2.update_layout(bargap=0.25)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No explicit NCO link column found in schemes to compute coverage.")

# -------------------------------
# Footer / Tips
# -------------------------------
st.markdown("---")
st.markdown(
    """
**Tips**
- For best voice accuracy, keep uploads 5–15 seconds, clear speech, and low background noise.
- If translation fails sporadically, ensure `googletrans==4.0.0-rc1`.
- You can enrich your scheme mapping by adding a precise NCO link column (e.g. `"NCO Code"`) in `govt_schemes.csv`.
- Want career guidance next? Plug an LLM on top of the retrieved NCO role + scheme context.
"""
)

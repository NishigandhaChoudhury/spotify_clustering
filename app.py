# app.py — Spotify Music Clustering App (Clean & User Friendly)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Music Personality",
    page_icon="🎵",
    layout="centered"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Circular+Std&display=swap');

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #0d0d0d 100%);
        color: white;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}

    /* Hero section */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #1DB954, #1ed760, #17a844);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #b3b3b3;
        font-size: 1.1rem;
        margin-top: 0;
    }

    /* Card */
    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
    }

    /* Cluster result box */
    .result-box {
        background: linear-gradient(135deg, #1DB954, #17a844);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin-top: 1.2rem;
    }
    .result-box h2 {
        color: white;
        font-size: 1.8rem;
        margin: 0;
    }
    .result-box p {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    /* Stat cards */
    .stat-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin: 1rem 0;
    }
    .stat-card {
        background: rgba(29, 185, 84, 0.12);
        border: 1px solid rgba(29, 185, 84, 0.3);
        border-radius: 12px;
        padding: 0.8rem 1.5rem;
        text-align: center;
        flex: 1;
    }
    .stat-card .number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1DB954;
    }
    .stat-card .label {
        font-size: 0.8rem;
        color: #b3b3b3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Feature badge */
    .feature-badge {
        display: inline-block;
        background: rgba(29,185,84,0.15);
        border: 1px solid rgba(29,185,84,0.4);
        color: #1DB954;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.85rem;
        margin: 0.2rem;
    }

    /* Section header */
    .section-title {
        color: white;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1DB954;
        padding-left: 0.8rem;
    }

    /* Slider labels */
    .stSlider label {
        color: #e0e0e0 !important;
        font-weight: 600;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1DB954, #17a844) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.7rem 2.5rem !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: scale(1.03);
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.1) !important;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL & DATA ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('kmeans_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv('clustered_songs.csv')

model, scaler = load_model()
df = load_data()
features = ['danceability', 'energy', 'tempo', 'loudness', 'valence']
n_clusters = df['cluster'].nunique()

# ─── CLUSTER PERSONALITIES ─────────────────────────────────────────────────────
cluster_info = {
    0: {"name": "Calm & Chill",        "emoji": "😌", "desc": "Soft, relaxing songs perfect for studying or winding down.", "color": "#4ECDC4"},
    1: {"name": "Danceable & Upbeat",  "emoji": "💃", "desc": "Feel-good tracks made for dancing and parties.",            "color": "#FF6B6B"},
    2: {"name": "High Energy",         "emoji": "🤘", "desc": "Intense, powerful songs for workouts and hype moments.",    "color": "#FFE66D"},
    3: {"name": "Sad & Emotional",     "emoji": "🌧️", "desc": "Deep, emotional music for introspective moods.",           "color": "#A8DADC"},
    4: {"name": "Happy & Vibrant",     "emoji": "☀️", "desc": "Bright, cheerful tunes that instantly lift your mood.",    "color": "#95E1D3"},
}

# ─── HERO SECTION ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎵 Music Personality Finder</h1>
    <p>Think of a song you love — find out what type of music person you are!</p>
</div>
""", unsafe_allow_html=True)

# ─── STATS ROW ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-row">
    <div class="stat-card">
        <div class="number">{len(df):,}</div>
        <div class="label">Songs Studied</div>
    </div>
    <div class="stat-card">
        <div class="number">{n_clusters}</div>
        <div class="label">Music Moods</div>
    </div>
    <div class="stat-card">
        <div class="number">5</div>
        <div class="label">Song Traits</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── FEATURES USED ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="card">
    <div class="section-title">🎛️ What We Look At In Every Song</div>
    <p style="color:#b3b3b3; font-size:0.9rem; margin-bottom:0.8rem;">
        Every song has 5 hidden scores that describe how it sounds and feels:
    </p>
    <span class="feature-badge">💃 How danceable it is</span>
    <span class="feature-badge">⚡ How energetic it feels</span>
    <span class="feature-badge">🥁 How fast or slow it is</span>
    <span class="feature-badge">🔊 How loud it is</span>
    <span class="feature-badge">😊 How happy or sad it sounds</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── CLUSTER EXPLORER ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🎭 The 5 Music Moods</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#b3b3b3; font-size:0.9rem;'>We grouped all songs into these 5 moods based on how they sound:</p>", unsafe_allow_html=True)

cols = st.columns(5)
for i, col in enumerate(cols):
    info = cluster_info.get(i, {"name": f"Cluster {i}", "emoji": "🎵", "desc": "", "color": "#1DB954"})
    with col:
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
                    border-top: 3px solid {info['color']}; border-radius:12px;
                    padding:0.8rem; text-align:center; height:130px;">
            <div style="font-size:1.8rem">{info['emoji']}</div>
            <div style="color:white; font-weight:700; font-size:0.8rem; margin-top:0.3rem">{info['name']}</div>
            <div style="color:#b3b3b3; font-size:0.7rem; margin-top:0.3rem">{info['desc'][:50]}...</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ─── PREDICT YOUR SONG ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🎯 What\'s Your Music Mood?</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#b3b3b3; font-size:0.9rem; margin-bottom:1.2rem;'>Think of a song you love right now. Move the sliders to describe how it feels, then hit the button!</p>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    dance = st.slider("💃 Can you dance to it?", 0.0, 1.0, 0.5, 0.01,
                      help="0 = no way, 1 = impossible not to dance")
    energy = st.slider("⚡ How energetic does it feel?", 0.0, 1.0, 0.5, 0.01,
                       help="0 = super calm and quiet, 1 = wild and intense")
    tempo = st.slider("🥁 How fast is it? (beats per min)", 50.0, 220.0, 120.0, 1.0,
                      help="60 = very slow, 120 = normal, 180 = very fast")

with col2:
    loudness = st.slider("🔊 How loud is it?", -60.0, 0.0, -10.0, 0.5,
                         help="-60 = very quiet, 0 = very loud")
    valence = st.slider("😊 Does it make you happy or sad?", 0.0, 1.0, 0.5, 0.01,
                        help="0 = sad/dark feeling, 1 = happy/uplifting feeling")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🎵 Find My Music Mood!"):
    input_data = np.array([[dance, energy, tempo, loudness, valence]])
    input_scaled = scaler.transform(input_data)
    cluster_pred = int(model.predict(input_scaled)[0])
    info = cluster_info.get(cluster_pred, {"name": "Unique Vibe", "emoji": "🎵", "desc": "A truly unique sound!", "color": "#1DB954"})

    st.markdown(f"""
    <div class="result-box" style="background: linear-gradient(135deg, {info['color']}cc, {info['color']}88);">
        <div style="font-size: 3rem">{info['emoji']}</div>
        <h2>You're a {info['name']} listener!</h2>
        <p>{info['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Your Song\'s Vibe Breakdown</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(7, 2.5))
    fig2.patch.set_facecolor('#1a1a2e')
    ax2.set_facecolor('#0f0f1a')

    feature_labels = ['Can dance to it', 'How energetic', 'How fast (speed)', 'How loud', 'Happy or sad']
    user_values = [dance, energy, tempo/220, (loudness+60)/60, valence]
    bar_colors = [info['color']] * 5

    bars = ax2.barh(feature_labels, user_values, color=bar_colors, alpha=0.85, height=0.5)
    ax2.set_xlim(0, 1)
    ax2.tick_params(colors='#b3b3b3', labelsize=8)
    ax2.set_xlabel('Score (Low → High)', color='#b3b3b3', fontsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#333')

    plt.tight_layout()
    st.pyplot(fig2)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#535353; font-size:0.8rem; padding-bottom:1rem;">
    Built with KMeans Clustering · Spotify Tracks Dataset · Powered by Streamlit
</div>
""", unsafe_allow_html=True)

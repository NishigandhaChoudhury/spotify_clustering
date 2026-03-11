# app.py — Streamlit App for Spotify Music Clustering

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Spotify Music Clustering",
    page_icon="🎵",
    layout="wide"
)

# ─── TITLE ─────────────────────────────────────────────────
st.title("🎵 Spotify Music Clustering")
st.markdown("Discover listening patterns by clustering songs based on audio features!")
st.markdown("---")

# ─── LOAD MODEL & DATA ─────────────────────────────────────
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

# ─── SIDEBAR ───────────────────────────────────────────────
st.sidebar.header("🎛️ Filter Songs")
selected_cluster = st.sidebar.selectbox(
    "Select a Cluster",
    options=['All'] + sorted(df['cluster'].unique().tolist())
)

# ─── MAIN LAYOUT ───────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total Songs", len(df))
col2.metric("Number of Clusters", df['cluster'].nunique())
col3.metric("Features Used", len(features))

st.markdown("---")

# ─── FILTERED DATA ─────────────────────────────────────────
if selected_cluster == 'All':
    filtered_df = df
else:
    filtered_df = df[df['cluster'] == int(selected_cluster)]

st.subheader(f"📊 Songs Data — {'All Clusters' if selected_cluster == 'All' else f'Cluster {selected_cluster}'}")
st.dataframe(filtered_df[features + ['cluster']].head(50), use_container_width=True)

# ─── PCA VISUALIZATION ────────────────────────────────────
st.markdown("---")
st.subheader("🔵 PCA Cluster Visualization")

X_scaled = scaler.transform(df[features])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
n_clusters = df['cluster'].nunique()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98FB98', '#F0E68C']

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(n_clusters):
    mask = df['cluster'] == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[i % len(colors)],
               label=f'Cluster {i}', alpha=0.5, s=5)
ax.set_title('Songs Clustered — PCA View')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.legend()
st.pyplot(fig)

# ─── CLUSTER AVERAGES ─────────────────────────────────────
st.markdown("---")
st.subheader("📈 Average Features per Cluster")
cluster_means = df.groupby('cluster')[features].mean().round(3)
st.dataframe(cluster_means, use_container_width=True)

fig2, ax2 = plt.subplots(figsize=(12, 4))
cluster_means.plot(kind='bar', ax=ax2, colormap='Set2')
ax2.set_title('Audio Features by Cluster')
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Average Value')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
plt.tight_layout()
st.pyplot(fig2)

# ─── PREDICT YOUR OWN SONG ────────────────────────────────
st.markdown("---")
st.subheader("🎯 Predict Your Song's Cluster!")
st.markdown("Move the sliders to enter a song's audio features:")

c1, c2, c3 = st.columns(3)
with c1:
    dance = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
with c2:
    tempo = st.slider("Tempo (BPM)", 50.0, 220.0, 120.0)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
with c3:
    valence = st.slider("Valence (Mood)", 0.0, 1.0, 0.5)

if st.button("🔍 Find My Cluster!", type="primary"):
    input_data = np.array([[dance, energy, tempo, loudness, valence]])
    input_scaled = scaler.transform(input_data)
    cluster_pred = model.predict(input_scaled)[0]

    cluster_info = {
        0: "😴 Calm & Relaxing",
        1: "💃 Danceable & Upbeat",
        2: "🤘 High Energy",
        3: "😢 Sad & Slow",
        4: "🎸 Energetic & Happy"
    }

    st.success(f"🎵 Your song belongs to **Cluster {cluster_pred}** — {cluster_info.get(cluster_pred, 'Unique Vibe!')}")

# ─── FOOTER ───────────────────────────────────────────────
st.markdown("---")
st.caption("Built with ❤️ using KMeans Clustering | Data: Spotify Tracks Dataset")
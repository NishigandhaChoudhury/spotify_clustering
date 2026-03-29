import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Spotify Music Clustering",
    page_icon="🎵",
    layout="wide"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark Spotify-like theme */
    .stApp { background-color: #121212; color: #FFFFFF; }
    .result-card {
        background: linear-gradient(135deg, #1DB954 0%, #158a3e 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
    }
    .mood-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: white;
        margin: 0;
    }
    .mood-subtitle {
        font-size: 1.1rem;
        color: rgba(255,255,255,0.85);
        margin-top: 8px;
    }
    .song-card {
        background: #1e1e1e;
        border-left: 4px solid #1DB954;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .song-title { font-weight: 700; font-size: 1rem; color: #FFFFFF; }
    .song-artist { color: #b3b3b3; font-size: 0.88rem; }
    .slider-hint { color: #b3b3b3; font-size: 0.78rem; margin-top: -12px; margin-bottom: 16px; }
    .share-box {
        background: #1e1e1e;
        border: 2px dashed #1DB954;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
    }
    .metric-card {
        background: #1e1e1e;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab"] { color: #b3b3b3; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #1DB954 !important; border-bottom: 3px solid #1DB954; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────────────────────────────
FEATURES = ['danceability', 'energy', 'tempo', 'loudness', 'valence']

CLUSTER_INFO = {
    0: {
        "name": "😤 Intense & Powerful",
        "desc": "You're drawn to raw energy and bold sounds. High intensity, low fluff.",
        "tags": ["#HighEnergy", "#Intense", "#Powerful"],
        "songs": [
            {"title": "Thunderstruck", "artist": "AC/DC"},
            {"title": "Killing in the Name", "artist": "Rage Against the Machine"},
            {"title": "Enter Sandman", "artist": "Metallica"},
        ]
    },
    1: {
        "name": "🌿 Chill & Acoustic",
        "desc": "Low-key, laid-back, introspective. You vibe to music that breathes.",
        "tags": ["#ChillVibes", "#Acoustic", "#Calm"],
        "songs": [
            {"title": "The Night We Met", "artist": "Lord Huron"},
            {"title": "Bloom", "artist": "The Paper Kites"},
            {"title": "Skinny Love", "artist": "Bon Iver"},
        ]
    },
    2: {
        "name": "🕺 Danceable & Upbeat",
        "desc": "You were born to move. High danceability, feel-good energy, pure fun.",
        "tags": ["#DanceFloor", "#PopVibes", "#Upbeat"],
        "songs": [
            {"title": "As It Was", "artist": "Harry Styles"},
            {"title": "Levitating", "artist": "Dua Lipa"},
            {"title": "Blinding Lights", "artist": "The Weeknd"},
        ]
    },
    3: {
        "name": "🎷 Mellow & Soulful",
        "desc": "Mood music. Rich tones, moderate pace — perfect for late nights.",
        "tags": ["#Soulful", "#Mellow", "#Jazzy"],
        "songs": [
            {"title": "Redbone", "artist": "Childish Gambino"},
            {"title": "Come Away With Me", "artist": "Norah Jones"},
            {"title": "Golden", "artist": "Jill Scott"},
        ]
    },
    4: {
        "name": "🔥 Electronic & Euphoric",
        "desc": "Festival energy, synth drops, unstoppable tempo. You live for the beat.",
        "tags": ["#Electronic", "#Euphoric", "#Festival"],
        "songs": [
            {"title": "Levels", "artist": "Avicii"},
            {"title": "One More Time", "artist": "Daft Punk"},
            {"title": "Titanium", "artist": "David Guetta"},
        ]
    },
}

SLIDER_HINTS = {
    "danceability": "0 = sitting completely still · 1 = can't stop moving",
    "energy":       "0 = slow & gentle · 1 = loud & intense",
    "valence":      "0 = sad & dark · 1 = happy & euphoric",
    "tempo":        "60 BPM = slow ballad · 180 BPM = fast EDM track",
    "loudness":     "-60 dB = nearly silent · 0 dB = maximally loud",
}

# ─── Load Data & Models ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("spotify_clustered.csv")

@st.cache_data
def load_pca_data():
    return pd.read_csv("pca_data.csv")

@st.cache_resource
def load_models():
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("pca_model.pkl", "rb") as f:
        pca = pickle.load(f)
    return kmeans, scaler, pca

try:
    df        = load_data()
    pca_df    = load_pca_data()
    kmeans, scaler, pca = load_models()
    models_loaded = True
except FileNotFoundError:
    models_loaded = False

# ─── Header ──────────────────────────────────────────────────────────────────────
st.markdown("# 🎵 Spotify Music Clustering")
st.markdown("Discover your music personality from **232,725 songs** · KMeans + PCA + t-SNE")
st.markdown("---")

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎚️ Find My Music Mood",
    "🔍 Cluster Visualization",
    "📊 Dataset Overview",
    "📈 Cluster Insights",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT (main tab, shown first)
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🎚️ Adjust the sliders to describe your music taste")
    st.caption("No typing needed — just slide and discover your music personality.")

    col1, col2 = st.columns(2)

    with col1:
        danceability = st.slider("💃 Danceability", 0.0, 1.0, 0.65, 0.01)
        st.markdown(f"<p class='slider-hint'>💡 {SLIDER_HINTS['danceability']}</p>", unsafe_allow_html=True)

        energy = st.slider("⚡ Energy", 0.0, 1.0, 0.70, 0.01)
        st.markdown(f"<p class='slider-hint'>💡 {SLIDER_HINTS['energy']}</p>", unsafe_allow_html=True)

        valence = st.slider("😊 Valence (Happiness)", 0.0, 1.0, 0.55, 0.01)
        st.markdown(f"<p class='slider-hint'>💡 {SLIDER_HINTS['valence']}</p>", unsafe_allow_html=True)

    with col2:
        tempo = st.slider("🥁 Tempo (BPM)", 50.0, 250.0, 120.0, 1.0)
        st.markdown(f"<p class='slider-hint'>💡 {SLIDER_HINTS['tempo']}</p>", unsafe_allow_html=True)

        loudness = st.slider("🔊 Loudness (dB)", -60.0, 0.0, -8.0, 0.5)
        st.markdown(f"<p class='slider-hint'>💡 {SLIDER_HINTS['loudness']}</p>", unsafe_allow_html=True)

    st.markdown("")
    predict_btn = st.button("🔍 Discover My Music Mood", type="primary", use_container_width=True)

    if predict_btn:
        if models_loaded:
            input_data   = np.array([[danceability, energy, tempo, loudness, valence]])
            input_scaled = scaler.transform(input_data)
            cluster_pred = int(kmeans.predict(input_scaled)[0])
            info         = CLUSTER_INFO.get(cluster_pred, CLUSTER_INFO[0])

            # ── Result Card ──────────────────────────────────────────────────
            st.markdown(f"""
            <div class="result-card">
                <p class="mood-title">{info['name']}</p>
                <p class="mood-subtitle">{info['desc']}</p>
                <p style="margin-top:14px; font-size:1rem; color:rgba(255,255,255,0.7);">
                    {'  '.join(info['tags'])}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Two Columns: Radar + Songs ───────────────────────────────────
            rc1, rc2 = st.columns([1, 1])

            with rc1:
                st.markdown("#### 🕸️ Your Music Personality Shape")

                # Normalize all 5 features to 0-1 for radar
                norm_vals = [
                    danceability,
                    energy,
                    valence,
                    (tempo - 50) / 200,
                    (loudness + 60) / 60,
                ]
                categories = ['Danceability', 'Energy', 'Valence', 'Tempo', 'Loudness']

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=norm_vals + [norm_vals[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    fillcolor='rgba(29,185,84,0.25)',
                    line=dict(color='#1DB954', width=3),
                    name='Your Profile'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='#1e1e1e',
                        radialaxis=dict(visible=True, range=[0, 1],
                                        tickfont=dict(color='#b3b3b3', size=10),
                                        gridcolor='#333'),
                        angularaxis=dict(tickfont=dict(color='white', size=12),
                                         gridcolor='#333')
                    ),
                    paper_bgcolor='#121212',
                    plot_bgcolor='#121212',
                    showlegend=False,
                    height=340,
                    margin=dict(t=20, b=20, l=40, r=40)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            with rc2:
                st.markdown("#### 🎵 Songs That Match Your Vibe")
                for song in info['songs']:
                    st.markdown(f"""
                    <div class="song-card">
                        <div class="song-title">🎵 {song['title']}</div>
                        <div class="song-artist">{song['artist']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Shareable Card ───────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📤 Share Your Music Mood")
            share_text = (
                f"🎵 My Spotify Music Mood: {info['name']}\n\n"
                f"{info['desc']}\n\n"
                f"{' '.join(info['tags'])}\n\n"
                f"My vibe → Danceability: {danceability:.2f} | Energy: {energy:.2f} | "
                f"Valence: {valence:.2f} | Tempo: {tempo:.0f} BPM | Loudness: {loudness:.1f} dB\n\n"
                f"Discover yours at: spotify-clustering.streamlit.app"
            )
            st.markdown(f"""
            <div class="share-box">
                <p style="font-size:1.5rem; margin:0;">{info['name']}</p>
                <p style="color:#b3b3b3; margin:8px 0 16px 0;">{info['desc']}</p>
                <p style="color:#1DB954; font-size:0.9rem;">📸 Screenshot this card and share it!</p>
            </div>
            """, unsafe_allow_html=True)

            st.code(share_text, language=None)
            st.caption("👆 Copy this text and paste it anywhere — Instagram caption, Twitter, WhatsApp!")

        else:
            st.error("⚠️ Model files not loaded. Make sure kmeans_model.pkl, scaler.pkl, pca_model.pkl are in your repo.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLUSTER VISUALIZATION
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    if models_loaded:
        st.subheader("🔵 PCA – How Songs Cluster Together")
        st.caption("Each dot is a song. Colors = different mood clusters. Similar songs sit closer together.")

        fig_pca = px.scatter(
            pca_df, x='PC1', y='PC2',
            color='cluster',
            color_continuous_scale='Viridis',
            labels={'cluster': 'Cluster', 'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
            title='PCA Visualization – Spotify Song Clusters',
            opacity=0.55,
            height=520,
        )
        fig_pca.update_traces(marker=dict(size=3))
        fig_pca.update_layout(paper_bgcolor='#121212', plot_bgcolor='#1e1e1e',
                              font=dict(color='white'))
        st.plotly_chart(fig_pca, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 How Many Songs Are in Each Cluster?")
        cluster_counts = df['cluster'].value_counts().sort_index().reset_index()
        cluster_counts.columns = ['Cluster', 'Songs']
        cluster_counts['Mood'] = cluster_counts['Cluster'].map(
            {k: v['name'] for k, v in CLUSTER_INFO.items()})

        fig_bar = px.bar(cluster_counts, x='Mood', y='Songs',
                         color='Songs', color_continuous_scale='Viridis',
                         title='Song Count per Mood Cluster', height=400)
        fig_bar.update_layout(paper_bgcolor='#121212', plot_bgcolor='#1e1e1e',
                               font=dict(color='white'))
        st.plotly_chart(fig_bar, use_container_width=True)

        try:
            c1, c2 = st.columns(2)
            c1.image("pca_clusters.png",  caption="PCA (Matplotlib)", use_container_width=True)
            c2.image("tsne_clusters.png", caption="t-SNE Visualization", use_container_width=True)
        except Exception:
            pass
    else:
        st.info("Load model files to see cluster visualizations.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    if models_loaded:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🎵 Total Songs",   f"{len(df):,}")
        col2.metric("🔢 Clusters",      df['cluster'].nunique())
        col3.metric("🎛️ Features Used", len(FEATURES))
        col4.metric("📊 Dataset",       "232,725 songs")

        st.markdown("---")
        st.subheader("📁 Dataset Preview (first 100 rows)")
        available = [f for f in FEATURES if f in df.columns]
        st.dataframe(df[available + ['cluster']].head(100), use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Feature Distributions")
        colors_list = ['#1DB954', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        fig, axes = plt.subplots(1, len(available), figsize=(16, 4))
        fig.patch.set_facecolor('#121212')
        for i, feat in enumerate(available):
            axes[i].set_facecolor('#1e1e1e')
            axes[i].hist(df[feat], bins=40, color=colors_list[i], alpha=0.9, edgecolor='#121212')
            axes[i].set_title(feat.capitalize(), fontsize=11, fontweight='bold', color='white')
            axes[i].tick_params(colors='white')
            axes[i].grid(alpha=0.2, color='gray')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Load model files to see dataset overview.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTER INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    if models_loaded:
        available = [f for f in FEATURES if f in df.columns]
        cluster_summary = df.groupby('cluster')[available].mean().round(3)

        st.subheader("🌡️ What Makes Each Cluster Unique?")
        st.caption("Brighter = higher average value for that feature in that cluster.")
        fig_heat = px.imshow(
            cluster_summary.T,
            color_continuous_scale='YlOrRd',
            text_auto=True,
            aspect='auto',
            title='Average Audio Feature per Cluster',
            height=380,
        )
        fig_heat.update_layout(paper_bgcolor='#121212', font=dict(color='white'))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Cluster Means")
        cluster_summary.index = [
            f"Cluster {i} — {CLUSTER_INFO.get(i, {}).get('name','')}"
            for i in cluster_summary.index
        ]
        st.dataframe(cluster_summary, use_container_width=True)

        st.markdown("---")
        st.subheader("🔗 Feature Correlation Matrix")
        st.caption("Do high-energy songs also have high danceability? Find out here.")
        corr = df[available].corr()
        fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                             title='Feature Correlation', height=420)
        fig_corr.update_layout(paper_bgcolor='#121212', font=dict(color='white'))
        st.plotly_chart(fig_corr, use_container_width=True)

        if 'genre_label' in df.columns:
            st.markdown("---")
            st.subheader("🎼 Top Genre per Cluster")
            top_genre = df.groupby('cluster')['genre_label'].agg(
                lambda x: x.value_counts().index[0]).reset_index()
            top_genre.columns = ['Cluster', 'Top Genre']
            top_genre['Mood'] = top_genre['Cluster'].map(
                {k: v['name'] for k, v in CLUSTER_INFO.items()})
            st.dataframe(top_genre[['Cluster', 'Mood', 'Top Genre']], use_container_width=True)
    else:
        st.info("Load model files to see cluster insights.")

# ─── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#b3b3b3; font-size:0.85rem;'>"
    "🎵 Spotify Music Clustering · 232,725 Songs · KMeans + PCA + t-SNE · Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)

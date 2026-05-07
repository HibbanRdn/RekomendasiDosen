import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import joblib
import scipy.sparse
from pathlib import Path

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rekomendasi Dosen Pembimbing",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #1a3a5c 0%, #2e6da4 100%);
        padding: 1.8rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .header-box h1 { margin: 0; font-size: 1.7rem; }
    .header-box p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }

    .result-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.7rem;
    }
    .score-bar-wrap { background:#e2e8f0; border-radius:6px; height:7px; margin-top:6px; }
    .score-bar      { height:7px; border-radius:6px;
                      background: linear-gradient(90deg,#2e6da4,#60a5fa); }
</style>
""", unsafe_allow_html=True)

# ─── Konstanta ────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

# ─── Validasi folder model ─────────────────────────────────────────────────────
REQUIRED_FILES = [
    "tfidf_vectorizer.joblib",
    "dosen_tfidf_matrix.npz",
    "dosen_embeddings.npy",
    "dosen_profiles.csv",
    "config.json",
    "indo_sbert_model",          # folder
]

def check_models():
    missing = [f for f in REQUIRED_FILES if not (MODEL_DIR / f).exists()]
    return missing

# ─── Load artefak (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Memuat TF-IDF vectorizer…")
def load_tfidf():
    vect   = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    matrix = scipy.sparse.load_npz(MODEL_DIR / "dosen_tfidf_matrix.npz")
    return vect, matrix

@st.cache_resource(show_spinner="Memuat SBERT embeddings & model…")
def load_sbert_artifacts():
    from sentence_transformers import SentenceTransformer
    embeddings = np.load(MODEL_DIR / "dosen_embeddings.npy")
    model      = SentenceTransformer(str(MODEL_DIR / "indo_sbert_model"))
    return model, embeddings

@st.cache_data
def load_profiles():
    return pd.read_csv(MODEL_DIR / "dosen_profiles.csv")

@st.cache_data
def load_config():
    with open(MODEL_DIR / "config.json") as f:
        return json.load(f)

# ─── Sastrawi (cached) ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Memuat stemmer bahasa Indonesia…")
def load_sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer  = StemmerFactory().create_stemmer()
    sw_remov = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, sw_remov

# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    stemmer, sw_remov = load_sastrawi()
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = sw_remov.remove(text)
    return stemmer.stem(text)

# ─── Fungsi rekomendasi ───────────────────────────────────────────────────────

def recommend_tfidf(judul: str, profiles, vect, matrix, top_k=5):
    from sklearn.metrics.pairwise import cosine_similarity
    qvec = vect.transform([preprocess(judul)])
    sims = cosine_similarity(qvec, matrix)[0]
    idxs = sims.argsort()[::-1][:top_k]
    return _build_df(idxs, profiles, score_vals=sims,
                     extra_cols=["Similarity"])

def recommend_sbert(judul: str, profiles, model, embeddings, top_k=5):
    from sklearn.metrics.pairwise import cosine_similarity
    qemb = model.encode([preprocess(judul)], convert_to_numpy=True)
    sims = cosine_similarity(qemb, embeddings)[0]
    idxs = sims.argsort()[::-1][:top_k]
    return _build_df(idxs, profiles, score_vals=sims,
                     extra_cols=["Similarity"])

def recommend_hybrid(judul: str, profiles, vect, matrix, model, embeddings,
                     top_k=5, alpha=0.4):
    from sklearn.metrics.pairwise import cosine_similarity

    clean = preprocess(judul)

    # TF-IDF
    qvec  = vect.transform([clean])
    st    = cosine_similarity(qvec, matrix)[0]
    mn, mx = st.min(), st.max()
    st_n  = (st - mn) / (mx - mn) if mx > mn else st

    # SBERT
    qemb  = model.encode([clean], convert_to_numpy=True)
    ss    = cosine_similarity(qemb, embeddings)[0]
    mn, mx = ss.min(), ss.max()
    ss_n  = (ss - mn) / (mx - mn) if mx > mn else ss

    hybrid = alpha * st_n + (1 - alpha) * ss_n
    idxs   = hybrid.argsort()[::-1][:top_k]

    rows = []
    for rank, i in enumerate(idxs, 1):
        rows.append({
            "Rank": rank,
            "Nama_Dosen":       profiles.iloc[i]["Nama_Dosen"],
            "Keahlian_Dasar":   profiles.iloc[i]["Keahlian_Dasar"],
            "Jumlah_Publikasi": int(profiles.iloc[i]["Jumlah_Publikasi"]),
            "Score_TFIDF":  round(float(st[i]), 4),
            "Score_SBERT":  round(float(ss[i]), 4),
            "Score_Hybrid": round(float(hybrid[i]), 4),
        })
    return pd.DataFrame(rows)

def _build_df(idxs, profiles, score_vals, extra_cols):
    rows = []
    for rank, i in enumerate(idxs, 1):
        row = {
            "Rank": rank,
            "Nama_Dosen":       profiles.iloc[i]["Nama_Dosen"],
            "Keahlian_Dasar":   profiles.iloc[i]["Keahlian_Dasar"],
            "Jumlah_Publikasi": int(profiles.iloc[i]["Jumlah_Publikasi"]),
        }
        for col in extra_cols:
            row[col] = round(float(score_vals[i]), 4)
        rows.append(row)
    return pd.DataFrame(rows)

# ─── Render kartu hasil ───────────────────────────────────────────────────────

RANK_COLORS = {1: "#f59e0b", 2: "#94a3b8", 3: "#cd7f32"}

def render_cards(df: pd.DataFrame, score_col: str, max_score: float = 1.0):
    for _, row in df.iterrows():
        r     = int(row["Rank"])
        color = RANK_COLORS.get(r, "#2e6da4")
        score = float(row[score_col])
        pct   = max(5, int(score / max_score * 100))

        st.markdown(f"""
        <div class="result-card">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="background:{color};color:white;border-radius:50%;
                        width:30px;height:30px;display:flex;align-items:center;
                        justify-content:center;font-weight:700;font-size:.9rem;
                        flex-shrink:0;">{r}</div>
            <div style="flex:1;">
              <div style="font-weight:600;font-size:.97rem;color:#1a3a5c;">
                {row['Nama_Dosen']}
              </div>
              <div style="color:#555;font-size:.82rem;margin-top:2px;">
                🔬 {row['Keahlian_Dasar']} &nbsp;·&nbsp; 📄 {int(row['Jumlah_Publikasi'])} publikasi
              </div>
            </div>
            <div style="text-align:right;min-width:65px;">
              <span style="font-size:1.1rem;font-weight:700;color:#1a3a5c;">{score:.4f}</span>
              <div style="color:#888;font-size:.72rem;">skor</div>
            </div>
          </div>
          <div class="score-bar-wrap">
            <div class="score-bar" style="width:{pct}%;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-box">
  <h1>🎓 Sistem Rekomendasi Dosen Pembimbing Skripsi</h1>
  <p>Program Studi Teknik Informatika — Universitas Lampung</p>
</div>
""", unsafe_allow_html=True)

# ── Cek keberadaan model ──
missing = check_models()
if missing:
    st.error("**Model belum ditemukan.** Pastikan folder `models/` ada di direktori yang sama dengan `app.py`.")
    st.markdown("File/folder yang kurang:")
    for m in missing:
        st.markdown(f"- `models/{m}`")
    st.info("Jalankan cell **Simpan Model** di notebook Anda terlebih dahulu, lalu salin folder `models/` ke sini.")
    st.stop()

# ── Load semua artefak ──
profiles          = load_profiles()
vect, tfidf_mat   = load_tfidf()
sbert_model, embs = load_sbert_artifacts()
config            = load_config()
best_alpha        = config["best_alpha"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameter")
    top_k = st.slider("Jumlah Rekomendasi (Top-K)", 1, 10, 5)

    st.markdown("---")
    st.markdown("### Metode Hybrid")
    alpha = st.slider(
        "Alpha (bobot TF-IDF)",
        0.0, 1.0, float(best_alpha), 0.05,
        help="Alpha = TF-IDF weight, (1−Alpha) = IndoBERT weight"
    )
    st.caption(f"TF-IDF **{alpha:.2f}** · IndoBERT **{1-alpha:.2f}**")
    st.caption(f"*(alpha optimal dari training: {best_alpha})*")

    st.markdown("---")
    st.markdown("### 📊 Info Dataset")
    st.metric("Jumlah Dosen", len(profiles))
    total_pub = profiles["Jumlah_Publikasi"].sum()
    st.metric("Total Publikasi", int(total_pub))

# ─── Input judul ──────────────────────────────────────────────────────────────
st.markdown("### ✏️ Masukkan Judul / Topik Skripsi")

CONTOH = [
    "— Pilih contoh —",
    "Penerapan Deep Learning untuk Deteksi Objek pada Video Pengawasan",
    "Rancang Bangun Sistem Informasi Manajemen Perpustakaan Berbasis Web",
    "Implementasi Internet of Things untuk Monitoring Kualitas Udara di Dalam Ruangan",
    "Klasifikasi Penyakit Tanaman Padi Menggunakan Convolutional Neural Network",
    "Analisis Sentimen Ulasan Produk Menggunakan IndoBERT",
    "Sistem Keamanan Jaringan Berbasis Machine Learning untuk Deteksi Intrusi",
]

col_a, col_b = st.columns([3, 2])
with col_b:
    pilihan = st.selectbox("Atau pilih contoh:", CONTOH)
with col_a:
    default = "" if pilihan == CONTOH[0] else pilihan
    judul   = st.text_area(
        "Judul atau topik skripsi:",
        value=default,
        height=95,
        placeholder="Contoh: Penerapan YOLO untuk Deteksi Kendaraan pada CCTV…",
    )

cari = st.button("🔍 Cari Rekomendasi", type="primary", use_container_width=True)

# ─── Hasil ────────────────────────────────────────────────────────────────────
if cari or judul.strip():
    if not judul.strip():
        st.warning("Silakan isi judul / topik skripsi terlebih dahulu.")
        st.stop()

    st.markdown("---")
    st.markdown(f"**Judul:** _{judul}_")
    st.markdown(f"**Top-K:** {top_k} rekomendasi")

    tab1, tab2, tab3 = st.tabs([
        "📐 TF-IDF + Cosine",
        "🤖 IndoBERT SBERT",
        f"⚡ Hybrid (α={alpha:.2f})",
    ])

    with tab1:
        with st.spinner("Menghitung TF-IDF…"):
            df1 = recommend_tfidf(judul, profiles, vect, tfidf_mat, top_k)
        render_cards(df1, "Similarity",
                     max_score=float(df1["Similarity"].max()))
        with st.expander("Lihat sebagai tabel"):
            st.dataframe(df1, use_container_width=True, hide_index=True)

    with tab2:
        with st.spinner("Menghitung IndoBERT SBERT…"):
            df2 = recommend_sbert(judul, profiles, sbert_model, embs, top_k)
        render_cards(df2, "Similarity",
                     max_score=float(df2["Similarity"].max()))
        with st.expander("Lihat sebagai tabel"):
            st.dataframe(df2, use_container_width=True, hide_index=True)

    with tab3:
        with st.spinner("Menghitung Hybrid…"):
            df3 = recommend_hybrid(judul, profiles, vect, tfidf_mat,
                                   sbert_model, embs, top_k, alpha)
        render_cards(df3, "Score_Hybrid",
                     max_score=float(df3["Score_Hybrid"].max()))
        with st.expander("Lihat sebagai tabel"):
            st.dataframe(df3[["Rank","Nama_Dosen","Keahlian_Dasar",
                               "Score_TFIDF","Score_SBERT","Score_Hybrid"]],
                         use_container_width=True, hide_index=True)

    # ── Download ──
    st.markdown("---")
    st.markdown("#### 💾 Unduh Hasil")
    c1, c2, c3 = st.columns(3)

    def to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    c1.download_button("⬇️ TF-IDF (.csv)",   to_csv(df1),
                       "rekomendasi_tfidf.csv",   "text/csv")
    c2.download_button("⬇️ IndoBERT (.csv)", to_csv(df2),
                       "rekomendasi_sbert.csv",   "text/csv")
    c3.download_button("⬇️ Hybrid (.csv)",   to_csv(df3),
                       "rekomendasi_hybrid.csv",  "text/csv")

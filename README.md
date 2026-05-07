# 🎓 Rekomendasi Dosen Pembimbing Skripsi

Sistem rekomendasi berbasis TF-IDF, IndoBERT Sentence-BERT, dan Hybrid.

## Struktur Folder

```
rekomendasi_dosen/
├── app.py                  ← Streamlit app
├── requirements.txt
├── models/                 ← Dihasilkan dari notebook
│   ├── tfidf_vectorizer.joblib
│   ├── dosen_tfidf_matrix.npz
│   ├── dosen_embeddings.npy
│   ├── dosen_profiles.csv
│   ├── config.json
│   └── indo_sbert_model/   ← folder model SBERT
```

## Langkah Deploy

### 1. Simpan model dari notebook
Tambahkan isi file `save_models.py` sebagai cell baru di akhir notebook,
lalu jalankan. Akan terbentuk folder `models/`.

### 2. Download folder `models/` dari Google Colab
Di Colab, jalankan:
```python
import shutil
shutil.make_archive('models', 'zip', 'models')
```
Kemudian download `models.zip` dari panel Files.

### 3. Jalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 4. Deploy ke Streamlit Community Cloud
1. Push ke GitHub (sertakan folder `models/`)
2. Buka https://share.streamlit.io → New app
3. Pilih repo & branch, main file: `app.py`
4. Deploy

> **Catatan:** Folder `indo_sbert_model/` bisa cukup besar (~400 MB).
> Untuk Streamlit Cloud, pastikan total repo < 1 GB (batas LFS GitHub).
> Alternatifnya, hapus `indo_sbert_model/` dan biarkan app download otomatis
> dari HuggingFace dengan mengganti baris load model menjadi:
> `SentenceTransformer("firqaaa/indo-sentence-bert-base")`

# Dokumentasi Teknis: Sistem Rekomendasi Dosen Pembimbing Skripsi
### Berbasis NLP Menggunakan TF-IDF, Cosine Similarity, dan Sentence-BERT

**Program Studi Teknik Informatika — Universitas Lampung**

**Disusun oleh:**
- Muhamad Hibban Ramadhan (2315061094)
- M. Sidiq Firdaus (23150610)
- Pangihutan Syahputra Purba (23150610)

---

## Daftar Isi

1. [Gambaran Umum Notebook](#1-gambaran-umum-notebook)
2. [Tahap 1 — Instalasi Library](#2-tahap-1--instalasi-library)
3. [Tahap 2 — Import Library](#3-tahap-2--import-library)
4. [Tahap 3 — Upload dan Load Data](#4-tahap-3--upload-dan-load-data)
5. [Tahap 4 — Data Cleaning](#5-tahap-4--data-cleaning)
6. [Tahap 5 — Text Preprocessing](#6-tahap-5--text-preprocessing)
7. [Tahap 6 — Pembuatan Profil Dosen](#7-tahap-6--pembuatan-profil-dosen)
8. [Tahap 7 — Metode 1: TF-IDF + Cosine Similarity (Baseline)](#8-tahap-7--metode-1-tf-idf--cosine-similarity-baseline)
9. [Tahap 8 — Metode 2: IndoBERT Sentence-BERT (Metode Utama)](#9-tahap-8--metode-2-indobert-sentence-bert-metode-utama)
10. [Tahap 9 — Evaluasi Top-K Accuracy](#10-tahap-9--evaluasi-top-k-accuracy)
11. [Tahap 10 — Perbandingan Dua Metode](#11-tahap-10--perbandingan-dua-metode)
12. [Tahap 10b — Metode 3: Hybrid TF-IDF + IndoBERT](#12-tahap-10b--metode-3-hybrid-tf-idf--indobert)
13. [Tahap 11 — Demo Sistem Rekomendasi](#13-tahap-11--demo-sistem-rekomendasi)
14. [Penyimpanan Model (Model Serialization)](#14-penyimpanan-model-model-serialization)
15. [Hubungan Antar Tahap dan Gambaran Pipeline Keseluruhan](#15-hubungan-antar-tahap-dan-gambaran-pipeline-keseluruhan)

---

## 1. Gambaran Umum Notebook

### 1.1 Konteks dan Latar Belakang

Notebook ini membangun sebuah **sistem rekomendasi dosen pembimbing skripsi** untuk mahasiswa Program Studi Teknik Informatika Universitas Lampung. Permasalahan yang diselesaikan adalah mismatch antara topik penelitian mahasiswa dan keahlian dosen pembimbing yang sering terjadi karena pemilihan pembimbing dilakukan secara manual atau berbasis pengetahuan yang terbatas.

Sistem ini mengotomasi proses pencocokan dengan cara membandingkan **representasi semantik judul penelitian mahasiswa** terhadap **profil akademik dosen** yang dibangun dari riwayat publikasinya.

### 1.2 Masalah yang Diselesaikan

Secara formal, masalah yang diselesaikan adalah:

> Diberikan judul atau topik skripsi mahasiswa sebagai query, temukan dosen pembimbing yang paling relevan berdasarkan kecocokan konten penelitian antara topik skripsi dengan riwayat publikasi akademik dosen.

### 1.3 Dataset yang Digunakan

Notebook ini bekerja dengan dua dataset:

**Dataset 1 — Data Usulan Skripsi (`data_usulan_skripsi_2026-05-07.csv`)**
Berisi data pengajuan judul skripsi mahasiswa, mencakup kolom seperti: NPM, Nama Mahasiswa, Judul Penelitian, Topik Penelitian, dan Dosen Pembimbing (yang menjadi ground truth evaluasi).

**Dataset 2 — Dataset Publikasi Dosen (`dataset_publikasi_dosen_unila_final.csv`)**
Berisi riwayat publikasi ilmiah dosen, mencakup kolom: Nama_Dosen, Judul_Publikasi, dan Keahlian_Dasar. Satu dosen dapat memiliki banyak baris karena setiap baris merepresentasikan satu judul publikasi.

### 1.4 Pendekatan dan Metodologi

Tiga pendekatan diimplementasikan dan dibandingkan secara sistematis:

| Metode | Teknik | Karakteristik |
|--------|--------|---------------|
| Metode 1 (Baseline) | TF-IDF + Cosine Similarity | Statistik berbasis frekuensi kata, tidak memahami konteks semantik |
| Metode 2 (Utama) | IndoBERT Sentence-BERT | Representasi semantik kontekstual berbasis Transformer |
| Metode 3 (Hybrid) | Weighted Average TF-IDF + IndoBERT | Menggabungkan keunggulan kedua metode sebelumnya |

### 1.5 Alur Besar Pipeline

```
Data Skripsi + Data Publikasi Dosen
         │
         ▼
  [4] Data Cleaning
  (Normalisasi nama, standarisasi topik)
         │
         ▼
  [5] Text Preprocessing
  (Lowercase → Cleaning → Stopword Removal → Stemming)
         │
         ▼
  [6] Pembuatan Profil Dosen
  (Agregasi semua publikasi per dosen → satu dokumen representatif)
         │
         ├──────────────────────────────┐
         ▼                              ▼
[7] TF-IDF Vectorization        [8] SBERT Encoding
    + Cosine Similarity              (IndoBERT)
         │                              │
         └──────────────┬───────────────┘
                        ▼
              [10b] Hybrid Scoring
                        │
                        ▼
              [9] Evaluasi Top-K Accuracy
                        │
                        ▼
              [11] Demo Sistem Rekomendasi
                        │
                        ▼
              [14] Penyimpanan Model
```

---

## 2. Tahap 1 — Instalasi Library

### Tujuan

Memastikan seluruh dependensi Python yang diperlukan tersedia di lingkungan Google Colab sebelum proses dimulai.

### Cell 1

```python
!pip install PySastrawi sentence-transformers scikit-learn pandas numpy matplotlib seaborn tqdm -q
```

**Penjelasan:**

Perintah `!pip install` dijalankan di dalam shell lingkungan Colab. Flag `-q` (quiet) menekan output instalasi yang panjang agar notebook tetap bersih. Library yang diinstal adalah:

- **PySastrawi**: Library NLP khusus bahasa Indonesia yang menyediakan stemmer (pemotong imbuhan) dan stopword remover berbasis kamus Bahasa Indonesia. Ini adalah library kritis karena tidak ada tool standar seperti NLTK atau spaCy yang secara native mendukung morfologi Bahasa Indonesia dengan baik.

- **sentence-transformers**: Library Python yang menyediakan antarmuka tingkat tinggi untuk memuatdan menggunakan model Sentence-BERT (SBERT), termasuk model berbasis IndoBERT yang digunakan dalam Metode 2. Library ini menangani encoding teks menjadi vektor dense secara efisien.

- **scikit-learn**: Library machine learning serbaguna yang digunakan untuk dua keperluan utama: `TfidfVectorizer` (membangun representasi TF-IDF) dan `cosine_similarity` (menghitung jarak antar vektor).

- **pandas**: Library manajemen data tabular. Digunakan intensif untuk membaca, membersihkan, mentransformasi, dan menyimpan dataset dalam format DataFrame.

- **numpy**: Library komputasi numerik. Digunakan untuk manipulasi array embedding dan operasi matriks.

- **matplotlib** dan **seaborn**: Digunakan untuk membuat visualisasi hasil evaluasi, termasuk bar chart perbandingan akurasi dan visualisasi distribusi topik.

- **tqdm**: Menyediakan progress bar saat memproses data dalam loop panjang, memberikan feedback visual tentang kemajuan komputasi.

---

## 3. Tahap 2 — Import Library

### Tujuan

Mengimpor semua modul ke dalam namespace Python dan mengonfigurasi preferensi tampilan global.

### Cell 2

```python
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.max_columns', 20)

print('Semua library berhasil diimport.')
```

**Penjelasan per modul:**

- `pandas as pd` dan `numpy as np`: Diimpor dengan alias standar untuk kemudahan penulisan kode.

- `re`: Modul regular expression bawaan Python, digunakan dalam fungsi preprocessing untuk membersihkan karakter non-alfabet dari teks.

- `matplotlib.pyplot as plt` dan `matplotlib.ticker as mtick`: `plt` untuk membuat dan merender plot, sedangkan `mtick` (khususnya `mtick.PercentFormatter()`) digunakan untuk memformat sumbu Y chart agar menampilkan tanda persen (`%`) secara otomatis.

- `seaborn as sns`: Diimpor meskipun tidak terlihat penggunaan eksplisitnya di cell-cell utama; kemungkinan digunakan untuk mengatur tema visual secara implisit atau untuk visualisasi tambahan.

- `from tqdm import tqdm`: Mengimpor kelas `tqdm` untuk membungkus iterator sehingga proses loop panjang menampilkan progress bar.

- `StemmerFactory` dan `StopWordRemoverFactory` dari Sastrawi: Factory pattern yang digunakan untuk membuat instance stemmer dan stopword remover. Penggunaan factory pattern memungkinkan PySastrawi menginisialisasi kamus internal secara terenkapsulasi.

- `TfidfVectorizer` dari scikit-learn: Kelas yang mengimplementasikan pipeline TF-IDF lengkap, dari tokenisasi hingga pembentukan matriks sparse.

- `cosine_similarity` dari scikit-learn: Fungsi yang menghitung cosine similarity antara dua matriks vektor secara efisien menggunakan operasi linear algebra.

- `SentenceTransformer`: Kelas utama dari library `sentence-transformers` untuk memuat model pretrained dan melakukan encoding kalimat menjadi vektor dense.

**Konfigurasi global:**

- `warnings.filterwarnings('ignore')`: Menekan warning Python yang tidak kritis (misalnya deprecation warning dari library) agar output notebook tetap bersih.

- `pd.set_option('display.max_colwidth', 80)`: Membatasi lebar kolom yang ditampilkan di Jupyter menjadi 80 karakter, mencegah tabel meluap secara horizontal.

- `pd.set_option('display.max_columns', 20)`: Membatasi jumlah kolom yang ditampilkan menjadi maksimal 20 kolom.

---

## 4. Tahap 3 — Upload dan Load Data

### Tujuan

Memuat kedua dataset (data skripsi mahasiswa dan dataset publikasi dosen) ke dalam memori sebagai DataFrame pandas, dan melakukan inspeksi awal untuk memahami struktur data.

### Cell 3 & 4 — Upload File

```python
from google.colab import files

print('Silakan upload file: data_usulan_skripsi_2026-05-07.csv')
uploaded_skripsi = files.upload()
```

```python
print('Silakan upload file: dataset_publikasi_dosen_unila_final.csv')
uploaded_dosen = files.upload()
```

**Penjelasan:**

Modul `google.colab.files` menyediakan antarmuka upload file interaktif dalam lingkungan Google Colab. Pemanggilan `files.upload()` memunculkan dialog pemilihan file dari perangkat lokal pengguna. File yang diupload kemudian tersedia di filesystem Colab (direktori kerja saat ini) dengan nama aslinya. Kedua cell dipisah agar pengguna dapat meng-upload satu file di satu waktu tanpa kebingungan.

### Cell 5 — Load Dataset

```python
df_skripsi = pd.read_csv('data_usulan_skripsi_2026-05-07.csv')
df_dosen   = pd.read_csv('dataset_publikasi_dosen_unila_final.csv')

print('--- Data Usulan Skripsi ---')
print(f'Jumlah baris  : {len(df_skripsi)}')
print(f'Jumlah kolom  : {len(df_skripsi.columns)}')
print(f'Kolom         : {df_skripsi.columns.tolist()}')

print('--- Dataset Publikasi Dosen ---')
print(f'Jumlah baris  : {len(df_dosen)}')
print(f'Jumlah kolom  : {len(df_dosen.columns)}')
print(f'Kolom         : {df_dosen.columns.tolist()}')
print(f'Jumlah dosen  : {df_dosen["Nama_Dosen"].nunique()}')
```

**Penjelasan:**

`pd.read_csv()` membaca file CSV dan menghasilkan DataFrame. Inspeksi awal menggunakan `len()`, `.columns.tolist()`, dan `.nunique()` memberikan gambaran tentang:

- Berapa banyak mahasiswa yang mengajukan skripsi.
- Kolom apa saja yang tersedia di setiap dataset.
- Berapa jumlah dosen unik dalam dataset publikasi. Angka ini penting karena menentukan jumlah kandidat rekomendasi yang tersedia.

Perlu dicatat bahwa `df_dosen` kemungkinan memiliki banyak baris per dosen (satu baris per judul publikasi), sehingga `len(df_dosen)` akan jauh lebih besar dari jumlah dosen unik (`df_dosen["Nama_Dosen"].nunique()`).

---

## 5. Tahap 4 — Data Cleaning

### Tujuan

Mengatasi ketidakkonsistenan dalam data mentah sebelum teks diproses lebih lanjut. Dua masalah utama yang ditangani adalah inkonsistensi penulisan nama dosen dan inkonsistensi penulisan topik penelitian.

### 5.1 Normalisasi Nama Dosen

#### Cell 6

```python
nama_mapping = {
    'Dr. Eng. Ir. Mardiana, S.T., M.T., IPM.' : 'Dr. Eng. Ir. Mardiana, M.T., IPM',
    'Ir. Gigih Forda Nama, S.T., M.T.I, IPM.' : 'Ir. Gigih Forda Nama, S.T., M.T.I., IPM',
    'Ir. Ing. Hery Dian Septama, S.T., IPM.'  : 'Ir. Ing. Hery Dian Septama, S.T., IPM',
    'Ir. Meizano Ardhi Muhammad, S.T., M.T., IPM.' : 'Ir. Meizano Ardhi Muhammad, S.T., M.T., IPM',
    'Ir. Trisya Septiana, ST.,MT., IPM.'       : 'Ir. Trisya Septiana, S.T., M.T., IPM',
    'Puput Budi Wintoro, S. Kom, M.T.I'        : 'Puput Budi Wintoro, S.Kom., M.T.I.',
    'Rio Ariestia Pradipta, S.Kom.,M.T.I.'     : 'Rio Ariestia Pradipta, S.Kom., M.T.I.',
}

df_skripsi['Dosen Pembimbing'] = df_skripsi['Dosen Pembimbing'].replace(nama_mapping)

dosen_skripsi   = set(df_skripsi['Dosen Pembimbing'].unique())
dosen_publikasi = set(df_dosen['Nama_Dosen'].unique())
tidak_cocok     = dosen_skripsi - dosen_publikasi
```

**Penjelasan mendalam:**

Masalah inti adalah bahwa data skripsi dan data publikasi dibuat secara independen, sehingga penulisan nama dosen tidak terstandardisasi. Contoh variasi yang ditemukan:

- Perbedaan penulisan gelar: `S. Kom` vs `S.Kom.` (spasi setelah S).
- Perbedaan titik pada singkatan: `M.T.I` vs `M.T.I.` (titik penutup).
- Perbedaan urutan gelar: `ST.,MT.` (tanpa titik dalam singkatan) vs `S.T., M.T.` (dengan titik setelah setiap huruf).
- Perbedaan kehadiran gelar tertentu: `Dr. Eng. Ir. Mardiana, S.T., M.T.` vs `Dr. Eng. Ir. Mardiana, M.T.` (gelar S.T. dihilangkan di dataset publikasi).

Normalisasi dilakukan dengan `dict.replace()` pandas yang menggantikan string secara exact-match. Setelah normalisasi, dilakukan **verifikasi** menggunakan operasi set difference (`dosen_skripsi - dosen_publikasi`) untuk memastikan tidak ada nama yang tertinggal tidak cocok. Ini adalah langkah quality assurance yang kritis karena jika nama tidak cocok, evaluasi sistem akan gagal (dosen pembimbing asli tidak akan pernah ditemukan dalam profil).

### 5.2 Standarisasi Topik Penelitian

#### Cell 7 — Fungsi `standardize_topic`

```python
def standardize_topic(topik):
    if not isinstance(topik, str):
        return 'Lainnya'
    t = topik.lower().strip()
    if any(k in t for k in ['machine learning', 'ml']):
        return 'Machine Learning'
    if any(k in t for k in ['deep learning']):
        return 'Deep Learning'
    if any(k in t for k in ['computer vision', 'pengolahan citra', 'image processing', 'citra']):
        return 'Computer Vision'
    # ... (dan seterusnya untuk semua kategori)
    return 'Lainnya'

df_skripsi['Topik_Asli']       = df_skripsi['Topik Penelitian']
df_skripsi['Topik Penelitian'] = df_skripsi['Topik Penelitian'].apply(standardize_topic)
```

**Penjelasan mendalam:**

Kolom `Topik Penelitian` di data skripsi mengandung nilai yang diisi secara bebas oleh mahasiswa, sehingga variasi sangat tinggi. Contoh: `RPL`, `Rekayasa Perangkat Lunak`, dan `rekayasa perangkat lunak` semuanya merujuk konsep yang sama.

Fungsi `standardize_topic` mengimplementasikan **rule-based classification** berbasis keyword matching:

1. **Type-checking**: Langkah pertama adalah memvalidasi bahwa input adalah string (`isinstance(topik, str)`). Nilai kosong atau NaN dikembalikan sebagai `'Lainnya'` untuk menghindari error.

2. **Case normalization**: Teks dikonversi ke huruf kecil (`topik.lower()`) sebelum dilakukan pencocokan, sehingga `Machine Learning`, `machine learning`, dan `MACHINE LEARNING` semua tertangkap.

3. **Keyword matching dengan `any()`**: Konstruksi `any(k in t for k in [...])` memeriksa apakah salah satu dari beberapa kata kunci ada dalam teks. Ini lebih fleksibel daripada exact-match karena satu topik bisa dinyatakan dengan berbagai cara.

4. **Hierarki aturan**: Aturan diurutkan dari yang paling spesifik (Deep Learning) ke yang paling umum (Machine Learning, AI). Urutan ini penting karena topik "deep learning" juga sering mengandung kata "learning", sehingga harus ditangkap lebih dulu oleh aturan Deep Learning.

5. **Kolom cadangan**: Nilai asli disimpan di kolom `Topik_Asli` sebelum ditimpa, sehingga proses standarisasi bersifat non-destruktif dan dapat diaudit.

Alasan pendekatan ini dipilih dibandingkan ML classifier adalah kesederhanaan dan transparansi. Karena jumlah data terbatas dan topik-topiknya memiliki kata kunci yang cukup distinktif, rule-based approach cukup efektif tanpa memerlukan data training tambahan.

---

## 6. Tahap 5 — Text Preprocessing

### Tujuan

Mengubah teks mentah berbahasa Indonesia menjadi representasi yang bersih dan terstandarisasi agar model NLP dapat bekerja secara optimal. Preprocessing menghilangkan noise linguistik (stopword, imbuhan, karakter khusus) sehingga perbandingan antar teks menjadi lebih bermakna.

### Cell 8 — Inisialisasi dan Definisi Fungsi

```python
stemmer_factory  = StemmerFactory()
stemmer          = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

def preprocess_text(text):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text
```

**Penjelasan setiap langkah preprocessing:**

**Inisialisasi PySastrawi:**
Object `stemmer` dan `stopword_remover` diinisialisasi **di luar** fungsi `preprocess_text` secara sengaja. Ini merupakan praktik penting karena inisialisasi PySastrawi melibatkan loading kamus yang besar ke memori, yang merupakan operasi mahal. Jika inisialisasi dilakukan di dalam fungsi, maka setiap pemanggilan fungsi akan memuat ulang kamus tersebut, menyebabkan bottleneck performa yang signifikan.

**Langkah 1 — Validasi Input:**
```python
if not isinstance(text, str) or text.strip() == '':
    return ''
```
Menangani kasus edge: nilai NaN dari pandas (bukan string) dan string kosong atau yang hanya berisi spasi. Mengembalikan string kosong daripada error mencegah pipeline berhenti di tengah jalan.

**Langkah 2 — Lowercasing:**
```python
text = text.lower()
```
Mengubah semua huruf menjadi huruf kecil. Ini memastikan bahwa kata seperti "Jaringan" dan "jaringan" diperlakukan identik. Lowercasing harus dilakukan sebelum cleaning karena beberapa karakter seperti huruf kapital memiliki representasi yang berbeda.

**Langkah 3 — Cleaning dengan Regular Expression:**
```python
text = re.sub(r'[^a-z\s]', ' ', text)
text = re.sub(r'\s+', ' ', text).strip()
```
Baris pertama menghapus semua karakter yang **bukan** huruf a-z dan bukan spasi (`\s`). Ini menghilangkan angka, tanda baca, tanda hubung, dan karakter khusus lainnya. Angka (seperti tahun dalam judul penelitian) dan tanda baca (seperti titik dua, koma) tidak membawa nilai semantik dan justru bisa mencemari representasi vektor.

Baris kedua menggabungkan beberapa spasi berurutan (yang timbul akibat penghapusan karakter) menjadi satu spasi, dan menghapus spasi di awal/akhir teks.

**Langkah 4 — Stopword Removal:**
```python
text = stopword_remover.remove(text)
```
PySastrawi menggunakan daftar stopword Bahasa Indonesia yang mencakup kata-kata fungsional seperti "yang", "dengan", "untuk", "pada", "di", "ke", "dari", dan lain sebagainya. Kata-kata ini sangat sering muncul dalam semua dokumen sehingga tidak memiliki daya pembeda (discriminative power). Menghapusnya meningkatkan kualitas representasi TF-IDF karena bobot TF-IDF akan lebih terkonsentrasi pada kata-kata bermakna (kata konten).

**Langkah 5 — Stemming:**
```python
text = stemmer.stem(text)
```
Stemmer PySastrawi mengimplementasikan algoritma **Enhanced Confix Stripping (ECS)** yang dirancang khusus untuk morfologi Bahasa Indonesia. Stemmer ini memotong awalan (prefix) seperti pe-, me-, ber-, di-, ke-, ter-, dan akhiran (suffix) seperti -kan, -an, -i, -nya sehingga "mengklasifikasikan", "klasifikasi", dan "diklasifikasi" semuanya menjadi "klasifikasi". Ini sangat penting untuk Bahasa Indonesia karena variasinya sangat kaya secara morfologis.

Urutan langkah (stopword removal **sebelum** stemming) dipilih secara sengaja karena beberapa stopword dalam Bahasa Indonesia bisa berubah bentuk setelah stemming, sehingga lebih baik menghapusnya dalam bentuk aslinya terlebih dahulu.

### Cell 9 — Contoh Uji Preprocessing

```python
contoh = [
    'Penerapan Arsitektur ConvNeXt untuk Deteksi Dini Tuberkolosis pada Citra X-ray Dada',
    'Rancang Bangun Sistem Informasi Manajemen Aset Berbasis Web Menggunakan Framework Laravel',
    'Klasifikasi Penyakit Tanaman Padi Menggunakan Convolutional Neural Network'
]
```

Cell ini berfungsi sebagai **sanity check** — memverifikasi secara visual bahwa fungsi preprocessing bekerja sebagaimana mestinya sebelum diterapkan ke seluruh dataset. Tiga contoh dipilih dari topik yang berbeda (Computer Vision, Web Development, Machine Learning) untuk memastikan fungsi berjalan baik untuk berbagai konteks.

### Cell 10 — Penerapan Preprocessing ke Seluruh Dataset

```python
df_skripsi['judul_clean'] = [
    preprocess_text(t) for t in tqdm(df_skripsi['Judul Penelitian'], desc='Skripsi')
]

df_dosen['judul_clean'] = [
    preprocess_text(t) for t in tqdm(df_dosen['Judul_Publikasi'], desc='Publikasi')
]
```

**Penjelasan:**

Preprocessing diterapkan ke kolom `Judul Penelitian` di `df_skripsi` dan kolom `Judul_Publikasi` di `df_dosen`, menghasilkan kolom baru `judul_clean` di masing-masing DataFrame. Penggunaan list comprehension dengan `tqdm` memberikan progress bar selama proses yang mungkin memakan waktu beberapa menit karena stemming PySastrawi relatif lambat per dokumen. Hasil disimpan dalam kolom baru (tidak menimpa kolom asli) untuk menjaga integritas data.

---

## 7. Tahap 6 — Pembuatan Profil Dosen

### Tujuan

Mengonstruksi representasi teks terpadu (unified textual representation) untuk setiap dosen, yang nantinya akan digunakan sebagai dokumen referensi dalam proses pencocokan kemiripan.

### Cell 11

```python
df_dosen['keahlian_clean'] = [
    preprocess_text(t) for t in df_dosen['Keahlian_Dasar']
]

dosen_profiles = (
    df_dosen
    .groupby('Nama_Dosen')
    .apply(lambda x: ' '.join(x['judul_clean'].tolist()) + ' ' + x['keahlian_clean'].iloc[0])
    .reset_index()
)
dosen_profiles.columns = ['Nama_Dosen', 'profile_text']

keahlian_asli = df_dosen[['Nama_Dosen', 'Keahlian_Dasar']].drop_duplicates()
dosen_profiles = dosen_profiles.merge(keahlian_asli, on='Nama_Dosen')

jml_publikasi  = df_dosen.groupby('Nama_Dosen').size().reset_index(name='Jumlah_Publikasi')
dosen_profiles = dosen_profiles.merge(jml_publikasi, on='Nama_Dosen')
```

**Penjelasan mendalam:**

**Konsep profil dosen:**
Dalam sistem ini, seorang dosen direpresentasikan oleh satu string teks panjang (`profile_text`) yang merupakan gabungan dari:
1. Semua judul publikasi yang sudah dipreprocess (konkatenasi).
2. Keahlian dasar yang sudah dipreprocess.

Ide di balik pendekatan ini adalah bahwa kumpulan judul publikasi seorang dosen mencerminkan bidang penelitian dan terminologi ilmiah yang sering ia gunakan. Semakin banyak seorang dosen mempublikasikan penelitian di bidang tertentu, semakin kuat representasi bidang tersebut dalam profilnya. Ini adalah **implicit expertise modeling** — keahlian dosen disimpulkan dari rekam jejaknya.

**Penambahan keahlian dasar:**
Keahlian dasar (`Keahlian_Dasar`) ditambahkan ke profil untuk mengatasi kasus di mana seorang dosen memiliki keahlian yang belum banyak tercermin dalam publikasinya (misalnya dosen baru atau dosen yang aktif di bidang pengajaran). Ini memberikan "prior" eksplisit tentang keahlian dosen.

**Operasi `groupby().apply()`:**
`groupby('Nama_Dosen')` mengelompokkan semua baris publikasi berdasarkan nama dosen. Fungsi lambda kemudian:
- Mengambil semua nilai kolom `judul_clean` dalam grup tersebut sebagai list.
- Menggabungkannya dengan `' '.join(...)` menjadi satu string.
- Menambahkan keahlian dasar dari baris pertama grup (`x['keahlian_clean'].iloc[0]`).

`.reset_index()` mengubah hasil groupby (yang menggunakan nama dosen sebagai index) kembali menjadi kolom biasa.

**Penambahan kolom metadata:**
Setelah profil dibangun, dua merge dilakukan untuk menambahkan informasi yang berguna untuk display:
- `Keahlian_Dasar` dalam bentuk aslinya (tidak dipreprocess) untuk keperluan tampilan yang readable.
- `Jumlah_Publikasi` yang merepresentasikan seberapa banyak rekam jejak dosen dalam dataset.

**Output:**
`dosen_profiles` adalah DataFrame dengan satu baris per dosen, berisi kolom: `Nama_Dosen`, `profile_text`, `Keahlian_Dasar`, dan `Jumlah_Publikasi`. Kolom `profile_text` inilah yang akan dikonsumsi oleh TF-IDF vectorizer maupun SBERT encoder di tahap berikutnya.

---

## 8. Tahap 7 — Metode 1: TF-IDF + Cosine Similarity (Baseline)

### Konsep TF-IDF

**TF-IDF (Term Frequency-Inverse Document Frequency)** adalah metode representasi teks berbasis statistik. Untuk setiap kata dalam dokumen, TF-IDF menghitung skor yang merupakan perkalian dari dua komponen:

- **TF (Term Frequency)**: Seberapa sering kata tersebut muncul dalam satu dokumen. Kata yang lebih sering muncul dalam dokumen mendapat skor TF lebih tinggi.

- **IDF (Inverse Document Frequency)**: Seberapa langka kata tersebut di seluruh koleksi dokumen. Kata yang muncul di banyak dokumen (misalnya "sistem", "berbasis") mendapat skor IDF rendah karena tidak memiliki daya pembeda yang kuat. Kata yang langka mendapat skor IDF tinggi.

Hasil akhir adalah vektor numerik sparse (mayoritas nilai nol) berukuran `n_fitur`, di mana setiap dimensi merepresentasikan satu kata atau n-gram dalam kosakata.

### Cell 12 — Pembangunan Matriks TF-IDF dan Fungsi Rekomendasi

```python
tfidf_vectorizer   = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
dosen_tfidf_matrix = tfidf_vectorizer.fit_transform(dosen_profiles['profile_text'])
```

**Penjelasan parameter:**

**`ngram_range=(1, 2)`:**
Parameter ini menentukan bahwa TF-IDF harus mengekstrak baik unigram (satu kata) maupun bigram (dua kata berurutan) sebagai fitur. Misalnya, dari teks "jaringan saraf tiruan" akan diekstrak: unigram `[jaringan, saraf, tiruan]` dan bigram `[jaringan saraf, saraf tiruan]`. Bigram sangat berguna karena banyak konsep dalam ilmu komputer hanya bermakna dalam pasangan kata, seperti "machine learning", "neural network", "sistem informasi". Jika hanya unigram yang digunakan, kata "machine" dan "learning" akan memiliki bobot tersendiri yang terlepas satu sama lain, padahal keduanya menjadi satu konsep.

Konsekuensi `ngram_range=(1, 2)`: Jumlah fitur (dimensi vektor) akan jauh lebih besar dibandingkan jika hanya menggunakan unigram, karena setiap pasangan kata unik menjadi satu fitur. Ini meningkatkan akurasi namun juga meningkatkan konsumsi memori.

**`min_df=1`:**
`min_df` (minimum document frequency) menentukan ambang batas minimal sebuah term harus muncul dalam berapa dokumen agar dimasukkan ke dalam kosakata. Nilai `1` berarti semua term yang muncul setidaknya di satu dokumen akan dimasukkan. Dalam konteks ini, karena jumlah dokumen (profil dosen) relatif sedikit, menggunakan nilai lebih tinggi seperti `2` atau `3` bisa mengeliminasi terlalu banyak term yang sebenarnya bermakna. Nilai `1` lebih aman untuk dataset kecil.

**`fit_transform()`:**
Operasi ini melakukan dua hal sekaligus: `.fit()` membangun kosakata dan menghitung statistik IDF dari semua profil dosen, kemudian `.transform()` mengonversi setiap profil menjadi vektor sparse. Hasilnya adalah matriks TF-IDF berukuran `(jumlah_dosen × jumlah_fitur)`.

**Fungsi `recommend_tfidf`:**

```python
def recommend_tfidf(judul_mahasiswa, top_k=5):
    judul_clean  = preprocess_text(judul_mahasiswa)
    query_vec    = tfidf_vectorizer.transform([judul_clean])
    similarities = cosine_similarity(query_vec, dosen_tfidf_matrix)[0]
    top_indices  = similarities.argsort()[::-1][:top_k]
    ...
```

**Alur logika fungsi:**

1. Preprocessing judul input mahasiswa menggunakan fungsi yang sama dengan preprocessing profil dosen. Ini kritis untuk menjamin konsistensi representasi — query dan dokumen harus berada dalam "bahasa" yang sama.

2. `.transform([judul_clean])` mengonversi query menjadi vektor TF-IDF menggunakan kosakata yang sudah dibangun dari profil dosen (bukan fit ulang). Penting: hanya `.transform()`, bukan `.fit_transform()`, karena kosakata sudah baku dari tahap fit.

3. `cosine_similarity(query_vec, dosen_tfidf_matrix)[0]` menghitung kemiripan cosine antara satu vektor query terhadap semua vektor profil dosen sekaligus. Hasilnya adalah array 1D berukuran `[0]` (diambil baris pertama dan satu-satunya) berisi skor kemiripan untuk setiap dosen.

4. `.argsort()[::-1][:top_k]` mengurutkan indeks berdasarkan skor secara descending (dari terbesar ke terkecil) dan mengambil `top_k` indeks pertama. `[::-1]` membalik urutan sort yang secara default ascending.

**Konsep Cosine Similarity:**

Cosine similarity mengukur sudut kosinus antara dua vektor. Nilai berkisar antara -1 hingga 1, namun karena TF-IDF selalu menghasilkan nilai non-negatif, praktisnya berkisar antara 0 hingga 1:
- Nilai mendekati **1**: Dua dokumen memiliki distribusi term yang sangat mirip (sudut kecil antar vektor).
- Nilai mendekati **0**: Dua dokumen tidak berbagi term yang signifikan (sudut 90 derajat).

Keunggulan cosine similarity dibandingkan Euclidean distance adalah tidak sensitif terhadap panjang dokumen — profil dosen yang memiliki 50 publikasi tidak secara otomatis "lebih dekat" ke semua query hanya karena vektornya lebih panjang.

---

## 9. Tahap 8 — Metode 2: IndoBERT Sentence-BERT (Metode Utama)

### Konsep BERT dan Sentence-BERT

**BERT (Bidirectional Encoder Representations from Transformers)** adalah model bahasa berbasis arsitektur Transformer yang memahami konteks kata secara bidirectional — setiap kata dipahami dalam konteks kata sebelum dan sesudahnya secara bersamaan. Ini berbeda fundamental dari model bag-of-words seperti TF-IDF yang tidak memperhitungkan urutan dan konteks kata.

**Sentence-BERT (SBERT)** adalah modifikasi BERT yang dioptimalkan untuk menghasilkan representasi vektor tingkat kalimat (sentence embeddings) yang bermakna secara semantik. Versi standar BERT kurang efisien untuk tugas kemiripan kalimat karena memerlukan input pasangan kalimat. SBERT menggunakan arsitektur Siamese/Triplet Network untuk menghasilkan embedding kalimat tunggal yang bisa langsung dibandingkan menggunakan cosine similarity.

**`firqaaa/indo-sentence-bert-base`** adalah model SBERT yang telah di-fine-tune khusus pada korpus bahasa Indonesia. Model ini berbasis IndoBERT (pretrained BERT untuk Bahasa Indonesia) yang kemudian di-fine-tune lebih lanjut untuk tugas sentence similarity menggunakan dataset paralel berbahasa Indonesia. Penggunaan model yang spesifik untuk Bahasa Indonesia sangat penting karena model berbasis bahasa Inggris tidak akan memahami morfologi dan semantik Bahasa Indonesia dengan baik.

### Cell 13 — Load Model

```python
model_sbert = SentenceTransformer('firqaaa/indo-sentence-bert-base')

print(f'Dimensi embedding : {model_sbert.get_sentence_embedding_dimension()}')
```

**Penjelasan:**

`SentenceTransformer()` mengunduh model dari Hugging Face Hub dan memuatnya ke memori. Proses ini membutuhkan waktu pada percobaan pertama karena ukuran model BERT base (sekitar 400-500 MB). Informasi dimensi embedding (biasanya 768 untuk model BERT base) penting untuk dipahami karena ini menentukan ukuran ruang vektor di mana representasi semantik berada.

### Cell 14 — Encoding Profil Dosen

```python
dosen_embeddings = model_sbert.encode(
    dosen_profiles['profile_text'].tolist(),
    batch_size=8,
    show_progress_bar=True,
    convert_to_numpy=True
)

print(f'Shape embedding dosen: {dosen_embeddings.shape}')
```

**Penjelasan parameter encoding:**

**`batch_size=8`:**
`batch_size` menentukan berapa banyak teks yang diproses sekaligus dalam satu batch oleh model BERT. Nilai `8` dipilih sebagai nilai moderat yang menyeimbangkan antara:
- Efisiensi komputasi: batch yang lebih besar mempercepat inferensi karena GPU dapat memproses beberapa input paralel.
- Keterbatasan memori: Profil dosen bisa sangat panjang (gabungan banyak judul publikasi), sehingga batch yang terlalu besar dapat menyebabkan Out-of-Memory error terutama pada GPU dengan VRAM terbatas (seperti Colab Free Tier).

Jika `batch_size` terlalu kecil (misalnya 1), pemrosesan menjadi lambat. Jika terlalu besar (misalnya 64), risiko OOM meningkat. Nilai 8 adalah pilihan konservatif yang aman untuk sebagian besar konfigurasi Colab.

**`convert_to_numpy=True`:**
Mengonversi output model dari PyTorch tensor menjadi NumPy array. Ini diperlukan karena `cosine_similarity` dari scikit-learn menerima NumPy array, bukan PyTorch tensor.

**`show_progress_bar=True`:**
Menampilkan progress bar tqdm selama proses encoding, memberikan feedback visual.

**Output:**
`dosen_embeddings` adalah matriks NumPy berukuran `(jumlah_dosen × dimensi_embedding)`. Setiap baris adalah representasi vektor dense profil seorang dosen dalam ruang semantik 768 dimensi.

**Fungsi `recommend_sbert`:**

```python
def recommend_sbert(judul_mahasiswa, top_k=5):
    judul_clean     = preprocess_text(judul_mahasiswa)
    query_embedding = model_sbert.encode([judul_clean], convert_to_numpy=True)
    similarities    = cosine_similarity(query_embedding, dosen_embeddings)[0]
    top_indices     = similarities.argsort()[::-1][:top_k]
    ...
```

Struktur fungsi ini identik dengan `recommend_tfidf` dalam alur logikanya, namun menggunakan representasi embedding neural network alih-alih vektor TF-IDF. Query dipreprocess, diencode menjadi vektor dense 768 dimensi, kemudian dibandingkan dengan semua embedding dosen menggunakan cosine similarity.

**Keunggulan SBERT atas TF-IDF:**

SBERT mampu menangkap kemiripan semantik yang tidak terdeteksi TF-IDF. Contoh: "pengolahan citra digital" dan "computer vision" memiliki kosakata yang berbeda sama sekali, tetapi dalam ruang embedding SBERT, keduanya akan memiliki vektor yang berdekatan karena model telah belajar bahwa keduanya merujuk konsep yang sama. TF-IDF akan memberikan similarity 0 untuk pasangan ini karena tidak ada term yang sama.

---

## 10. Tahap 9 — Evaluasi Top-K Accuracy

### Konsep Evaluasi

Evaluasi dilakukan menggunakan **ground truth** dari kolom `Dosen Pembimbing` di dataset skripsi — yaitu dosen yang benar-benar menjadi pembimbing skripsi tersebut. Asumsi yang digunakan adalah bahwa dosen pembimbing yang sesungguhnya adalah rekomendasi yang "benar" (relevan).

**Top-K Accuracy** mengukur: dari seluruh kasus evaluasi, berapa persen yang dosen pembimbingnya berhasil masuk dalam K rekomendasi teratas sistem? Ini adalah metrik yang umum digunakan dalam sistem rekomendasi dan information retrieval.

### Cell 15 — Filtering Data Evaluasi

```python
dosen_valid   = set(dosen_profiles['Nama_Dosen'].tolist())
df_eval       = df_skripsi[df_skripsi['Dosen Pembimbing'].isin(dosen_valid)].copy()
df_tidak_eval = df_skripsi[~df_skripsi['Dosen Pembimbing'].isin(dosen_valid)]
```

**Penjelasan:**

Hanya data skripsi yang dosen pembimbingnya **ada dalam dataset publikasi** yang dapat dievaluasi. Data yang dosen pembimbingnya tidak ada dalam dataset publikasi tidak dapat dievaluasi karena sistem tidak memiliki profil untuk dosen tersebut. Filtering ini penting untuk integritas evaluasi — memasukkan data yang tidak evaluable hanya akan menurunkan akurasi secara artifisial.

`.copy()` digunakan untuk membuat salinan independen DataFrame agar modifikasi selanjutnya tidak memengaruhi `df_skripsi` asli (menghindari `SettingWithCopyWarning` pandas).

### Cell 16 — Fungsi Evaluasi Generik

```python
def evaluate_model(recommend_func, df_eval, k_values=[1, 3, 5], nama_metode='Model'):
    max_k   = max(k_values)
    results = {k: 0 for k in k_values}
    detail  = []

    for _, row in tqdm(df_eval.iterrows(), total=len(df_eval), desc=nama_metode):
        judul_input  = row['Judul Penelitian']
        dosen_aktual = row['Dosen Pembimbing']

        rekomendasi  = recommend_func(judul_input, top_k=max_k)
        nama_rekomen = rekomendasi['Nama_Dosen'].tolist()

        row_result = {...}

        for k in k_values:
            hit = dosen_aktual in nama_rekomen[:k]
            row_result[f'Hit@{k}'] = hit
            if hit:
                results[k] += 1

        detail.append(row_result)

    total    = len(df_eval)
    accuracy = {k: round(v / total * 100, 2) for k, v in results.items()}
    ...
```

**Penjelasan mendalam:**

**Parameter `k_values=[1, 3, 5]`:**
Tiga nilai K yang dievaluasi memberikan gambaran lengkap tentang performa sistem di berbagai skenario penggunaan:
- **K=1 (Top-1 Accuracy)**: Mengukur seberapa sering rekomendasi pertama (paling tinggi) adalah dosen yang tepat. Ini standar paling ketat — sistem harus benar-benar presisi.
- **K=3 (Top-3 Accuracy)**: Dalam konteks nyata, mahasiswa mungkin diberikan 3 pilihan dosen. Top-3 mengukur seberapa sering dosen yang tepat ada di antara 3 pilihan tersebut.
- **K=5 (Top-5 Accuracy)**: Standar paling longgar, mengukur apakah dosen yang tepat paling tidak masuk dalam 5 rekomendasi. Ini relevan jika sistem digunakan sebagai "daftar kandidat" yang kemudian dipilih manusia.

Secara alami, Top-5 Accuracy selalu ≥ Top-3 Accuracy ≥ Top-1 Accuracy.

**Mekanisme evaluasi:**
Untuk setiap data skripsi dalam `df_eval`:
1. Sistem menghasilkan `max_k` (5) rekomendasi untuk judul skripsi tersebut.
2. Untuk setiap nilai K, dicek apakah nama dosen pembimbing aktual ada dalam K rekomendasi teratas (`hit = dosen_aktual in nama_rekomen[:k]`).
3. Jika ada (hit = True), counter untuk K tersebut ditambah 1.

**Efisiensi:** Fungsi hanya memanggil `recommend_func` sekali per data dengan `top_k=max(k_values)` (yaitu 5), kemudian evaluasi untuk K=1, K=3, dan K=5 dilakukan dari hasil yang sama. Ini lebih efisien daripada memanggil tiga kali.

**Output detail:**
Selain akurasi agregat, fungsi menghasilkan `detail` DataFrame yang berisi informasi hit/miss per mahasiswa. Ini berguna untuk analisis error lebih lanjut dan visualisasi akurasi per dosen.

### Cell 17 & 18 — Menjalankan Evaluasi

```python
acc_tfidf, detail_tfidf = evaluate_model(
    recommend_func = recommend_tfidf,
    df_eval        = df_eval,
    k_values       = [1, 3, 5],
    nama_metode    = 'TF-IDF + Cosine Similarity'
)

acc_sbert, detail_sbert = evaluate_model(
    recommend_func = recommend_sbert,
    df_eval        = df_eval,
    k_values       = [1, 3, 5],
    nama_metode    = 'IndoBERT Sentence-BERT'
)
```

**Penjelasan:**

Fungsi `evaluate_model` dipanggil dua kali dengan fungsi rekomendasi yang berbeda sebagai argumen. Desain ini (higher-order function / function as parameter) memungkinkan kode evaluasi yang sama digunakan untuk semua metode tanpa duplikasi — ini adalah penerapan prinsip DRY (Don't Repeat Yourself) yang baik.

---

## 11. Tahap 10 — Perbandingan Dua Metode

### Tujuan

Membandingkan hasil evaluasi TF-IDF dan SBERT secara tabular dan visual untuk menarik kesimpulan tentang metode mana yang lebih unggul dan dalam kondisi apa.

### Cell 19 — Tabel Perbandingan

```python
df_comparison = pd.DataFrame({
    'Metode'    : ['TF-IDF + Cosine Similarity', 'IndoBERT Sentence-BERT'],
    'Top-1 (%)' : [acc_tfidf[1], acc_sbert[1]],
    'Top-3 (%)' : [acc_tfidf[3], acc_sbert[3]],
    'Top-5 (%)' : [acc_tfidf[5], acc_sbert[5]],
})
```

**Penjelasan:**

DataFrame perbandingan dibangun langsung dari dictionary hasil evaluasi. Ini memberikan representasi tabular yang mudah dibaca dan dapat diekspor ke CSV.

### Cell 20 — Visualisasi 1: Grouped Bar Chart

```python
fig, ax = plt.subplots(figsize=(9, 5))

x         = np.arange(3)
width     = 0.35
labels_k  = ['Top-1', 'Top-3', 'Top-5']

bars1 = ax.bar(x - width/2, vals_tfidf, width, label='TF-IDF + Cosine Similarity',
               color='#4C72B0', edgecolor='white')
bars2 = ax.bar(x + width/2, vals_sbert, width, label='IndoBERT Sentence-BERT',
               color='#DD8452', edgecolor='white')
```

**Penjelasan:**

`x = np.arange(3)` membuat array `[0, 1, 2]` yang merepresentasikan posisi 3 grup bar (Top-1, Top-3, Top-5) pada sumbu X.

`width = 0.35` adalah lebar setiap bar. Karena ada dua bar per grup, mereka ditempatkan di posisi `x - width/2` dan `x + width/2` sehingga terpusat di masing-masing grup tanpa saling tumpang tindih.

Anotasi nilai persentase ditambahkan di atas setiap bar menggunakan `ax.text()` untuk memudahkan pembacaan nilai eksak tanpa harus memperkirakan dari gridline.

`ax.yaxis.set_major_formatter(mtick.PercentFormatter())` memformat label sumbu Y menjadi persentase (menambahkan tanda `%`).

`dpi=150` saat menyimpan menghasilkan gambar dengan resolusi yang cukup baik untuk laporan atau presentasi.

### Cell 21 — Visualisasi 2: Akurasi per Dosen

```python
def acc_per_dosen(detail_df, k=5):
    col     = f'Hit@{k}'
    grouped = detail_df.groupby('Dosen Aktual')[col].agg(['sum', 'count'])
    grouped['accuracy'] = grouped['sum'] / grouped['count'] * 100
    grouped = grouped.sort_values('accuracy', ascending=True)
    return grouped
```

**Penjelasan:**

Fungsi ini menghitung akurasi per dosen (bukan akurasi keseluruhan). Ini memberikan wawasan yang lebih granular: dosen mana yang mudah direkomendasikan dengan tepat, dan dosen mana yang sering terlewat oleh sistem.

`agg(['sum', 'count'])` menghitung dua statistik sekaligus: `sum` adalah jumlah hit (prediksi benar), `count` adalah total data dengan dosen tersebut sebagai pembimbing. Akurasi per dosen kemudian dihitung sebagai `sum / count * 100`.

Visualisasi menggunakan **horizontal bar chart** (`barh`) karena nama dosen panjang dan lebih mudah dibaca secara horizontal. `sharey=True` pada dua subplot memastikan sumbu Y (nama dosen) identik di kedua chart untuk kemudahan perbandingan langsung.

### Cell 22 — Visualisasi 3: Distribusi Topik

```python
topik_counts = df_eval['Topik Penelitian'].value_counts().sort_values(ascending=True)
ax.barh(topik_counts.index, topik_counts.values, color='#55A868', edgecolor='white')
```

**Penjelasan:**

Distribusi topik memberikan konteks tentang komposisi data evaluasi. Jika data evaluasi sangat tidak seimbang (misalnya 80% topik Machine Learning), maka akurasi sistem secara keseluruhan akan sangat dipengaruhi oleh seberapa baik sistem bekerja untuk topik Machine Learning saja. Visualisasi ini penting untuk memahami representatifitas hasil evaluasi.

---

## 12. Tahap 10b — Metode 3: Hybrid TF-IDF + IndoBERT

### Motivasi dan Konsep

Metode hybrid dirancang untuk memanfaatkan **komplementaritas** antara TF-IDF dan SBERT:

- **TF-IDF** unggul dalam mencocokkan kata kunci yang spesifik dan teknis. Jika judul skripsi mengandung kata "Arduino" dan profil dosen juga banyak mengandung "Arduino", TF-IDF akan memberikan skor tinggi.

- **SBERT** unggul dalam menangkap kemiripan semantik. Jika judul skripsi berbicara tentang "deteksi objek real-time" dan profil dosen berbicara tentang "computer vision untuk surveillance", SBERT bisa mendeteksi kemiripan ini meski tidak ada kata yang sama.

Hybrid menggabungkan keunggulan keduanya melalui **weighted average**.

### Cell 23 — Fungsi `recommend_hybrid`

```python
def recommend_hybrid(judul_mahasiswa, top_k=5, alpha=0.6):
    judul_clean = preprocess_text(judul_mahasiswa)

    # Skor TF-IDF dan normalisasi
    query_tfidf       = tfidf_vectorizer.transform([judul_clean])
    scores_tfidf      = cosine_similarity(query_tfidf, dosen_tfidf_matrix)[0]
    min_t, max_t      = scores_tfidf.min(), scores_tfidf.max()
    if max_t - min_t > 0:
        scores_tfidf_norm = (scores_tfidf - min_t) / (max_t - min_t)
    else:
        scores_tfidf_norm = scores_tfidf

    # Skor SBERT dan normalisasi
    query_embed       = model_sbert.encode([judul_clean], convert_to_numpy=True)
    scores_sbert      = cosine_similarity(query_embed, dosen_embeddings)[0]
    min_s, max_s      = scores_sbert.min(), scores_sbert.max()
    if max_s - min_s > 0:
        scores_sbert_norm = (scores_sbert - min_s) / (max_s - min_s)
    else:
        scores_sbert_norm = scores_sbert

    # Kombinasi dengan bobot
    scores_hybrid = alpha * scores_tfidf_norm + (1 - alpha) * scores_sbert_norm
```

**Penjelasan mendalam:**

**Mengapa normalisasi diperlukan?**
Skor TF-IDF dan skor SBERT memiliki distribusi yang berbeda secara intrinsik. Skor TF-IDF cosine similarity biasanya bernilai sangat kecil (misalnya antara 0.0 dan 0.15 untuk sebagian besar dokumen), sementara skor SBERT cosine similarity biasanya bernilai lebih tinggi (misalnya antara 0.5 dan 0.9 karena embedding dense memiliki similarity yang secara alami lebih tinggi). Jika keduanya digabungkan langsung tanpa normalisasi, skor SBERT akan selalu mendominasi karena nilainya lebih besar secara absolut, menjadikan bobot `alpha` tidak berarti.

**Normalisasi Min-Max:**
```
score_normalized = (score - min) / (max - min)
```
Normalisasi ini memetakan skor ke rentang [0, 1] untuk setiap metode secara independen, di mana skor terendah untuk query tersebut menjadi 0 dan skor tertinggi menjadi 1. Ini memungkinkan perbandingan dan penggabungan yang adil.

**Perlindungan terhadap division by zero:**
```python
if max_t - min_t > 0:
    ...
else:
    scores_tfidf_norm = scores_tfidf
```
Jika semua dosen memiliki skor yang identik (max - min = 0), normalisasi akan menghasilkan division by zero. Kondisi ini ditangani dengan menggunakan skor asli sebagai fallback.

**Parameter `alpha` dan tradeoffnya:**

`alpha` adalah bobot yang menentukan kontribusi relatif TF-IDF terhadap skor akhir:
- `scores_hybrid = alpha * scores_tfidf_norm + (1 - alpha) * scores_sbert_norm`
- Jika `alpha = 1.0`: Hybrid identik dengan TF-IDF murni.
- Jika `alpha = 0.0`: Hybrid identik dengan SBERT murni.
- Jika `alpha = 0.6`: TF-IDF berkontribusi 60% dan SBERT berkontribusi 40%.

Default `alpha = 0.6` dipilih sebagai starting point eksperimen yang memberikan bobot lebih besar ke TF-IDF. Nilai optimal dicari melalui grid search pada Cell berikutnya.

### Cell 24 — Grid Search Nilai Alpha Optimal

```python
alphas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
```

**Penjelasan:**

Grid search dilakukan dengan mengevaluasi 6 nilai alpha yang berbeda dalam rentang 0.3 hingga 0.8. Rentang ini dipilih karena:
- Nilai di bawah 0.3 berarti TF-IDF berkontribusi kurang dari 30%, hampir mengabaikan informasi keyword.
- Nilai di atas 0.8 berarti SBERT berkontribusi kurang dari 20%, hampir mengabaikan informasi semantik.
- Rentang 0.3–0.8 adalah zona yang masuk akal untuk eksplorasi awal.

Untuk setiap nilai alpha, evaluasi Top-K Accuracy dilakukan pada seluruh data evaluasi menggunakan pendekatan yang sama dengan Cell 17 dan 18. Alpha yang menghasilkan Top-5 Accuracy tertinggi dipilih sebagai `best_alpha`.

**Catatan penting tentang validitas evaluasi:**
Secara ideal, pencarian alpha optimal sebaiknya dilakukan pada **validation set** yang terpisah dari test set untuk menghindari overfitting terhadap data evaluasi. Dalam notebook ini, alpha dicari dan dievaluasi pada set data yang sama, sehingga ada risiko bahwa alpha yang terpilih sedikit overfit terhadap dataset yang tersedia. Namun, untuk dataset skala kecil seperti ini, praktik ini masih cukup dapat diterima.

### Cell 25 — Evaluasi Final Hybrid

```python
def recommend_hybrid_best(judul, top_k=5):
    return recommend_hybrid(judul, top_k=top_k, alpha=best_alpha)

acc_hybrid, detail_hybrid = evaluate_model(
    recommend_func = recommend_hybrid_best,
    df_eval        = df_eval,
    k_values       = [1, 3, 5],
    nama_metode    = f'Hybrid (alpha={best_alpha})'
)
```

**Penjelasan:**

Fungsi wrapper `recommend_hybrid_best` dibuat untuk "membekukan" nilai `best_alpha` yang ditemukan dari grid search, sehingga fungsi ini bisa dipass ke `evaluate_model` dengan signature yang konsisten `(judul, top_k=5)`. Ini adalah pola closure/partial application yang elegan.

### Cell 26 & 27 — Tabel dan Visualisasi Tiga Metode

**Visualisasi terdiri dari dua chart berdampingan:**

**Chart kiri (Grouped Bar):**
Membandingkan tiga metode pada tiga nilai K. Setiap grup berisi tiga bar (satu per metode). Lebar bar diubah menjadi `0.25` (dari `0.35` sebelumnya) untuk mengakomodasi tiga bar dalam satu grup tanpa saling tumpang tindih.

**Chart kanan (Line Chart pengaruh alpha):**
```python
ax2.plot(df_alpha['alpha'], df_alpha['top1'], marker='o', label='Top-1', ...)
ax2.plot(df_alpha['alpha'], df_alpha['top3'], marker='s', label='Top-3', ...)
ax2.plot(df_alpha['alpha'], df_alpha['top5'], marker='^', label='Top-5', ...)
ax2.axvline(x=best_alpha, color='gray', linestyle='--', ...)
```

Chart ini memvisualisasikan hasil grid search alpha — bagaimana akurasi berubah seiring perubahan bobot alpha. Garis vertikal putus-putus (`axvline`) menandai posisi alpha optimal yang terpilih. Chart ini memberikan pemahaman tentang sensitivitas model terhadap perubahan alpha: jika kurva relatif datar, model tidak terlalu sensitif terhadap nilai alpha; jika ada puncak yang tajam, pemilihan alpha kritis.

---

## 13. Tahap 11 — Demo Sistem Rekomendasi

### Tujuan

Mendemonstrasikan penggunaan sistem dalam skenario nyata dengan berbagai contoh judul skripsi dari topik yang berbeda.

### Cell 29 — Fungsi `demo_rekomendasi`

```python
def demo_rekomendasi(judul_input, top_k=5):
    print('=' * 80)
    print('SISTEM REKOMENDASI DOSEN PEMBIMBING SKRIPSI')
    ...
    hasil_tfidf  = recommend_tfidf(judul_input, top_k=top_k)
    hasil_sbert  = recommend_sbert(judul_input, top_k=top_k)
    hasil_hybrid = recommend_hybrid(judul_input, top_k=top_k, alpha=best_alpha)
```

**Penjelasan:**

Fungsi ini mengintegrasikan ketiga metode dalam satu tampilan output yang rapi dan informatif. Output disusun secara berurutan (TF-IDF → SBERT → Hybrid) dengan header yang jelas.

Untuk Metode Hybrid, kolom `Score_TFIDF`, `Score_SBERT`, dan `Score_Hybrid` ditampilkan bersama untuk memberikan transparansi tentang bagaimana skor akhir dikompositkan dari dua komponen.

### Cell 30, 31, 32 — Contoh Demo

```python
judul_demo  = 'Penerapan Deep Learning untuk Deteksi Objek pada Video Pengawasan'
judul_demo2 = 'Rancang Bangun Sistem Informasi Manajemen Perpustakaan Berbasis Web'
judul_demo3 = 'Implementasi Internet of Things untuk Monitoring Kualitas Udara di Dalam Ruangan'
```

**Penjelasan:**

Tiga judul demo dipilih dari topik yang berbeda secara sengaja:
- **Judul 1** (Deep Learning / Computer Vision): Menguji kemampuan sistem untuk topik AI/ML.
- **Judul 2** (Sistem Informasi / Web): Menguji untuk topik rekayasa perangkat lunak.
- **Judul 3** (IoT): Menguji untuk topik hardware dan embedded systems.

Variasi topik ini penting untuk memverifikasi bahwa sistem bekerja secara konsisten di seluruh domain, bukan hanya untuk satu topik tertentu.

---

## 14. Penyimpanan Model (Model Serialization)

### Tujuan

Menyimpan semua artefak model yang telah dilatih ke disk agar dapat digunakan kembali untuk deployment tanpa perlu melatih ulang dari awal.

### Cell 33

```python
import joblib
import scipy.sparse

SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, f"{SAVE_DIR}/tfidf_vectorizer.joblib")

# 2. TF-IDF matrix (sparse)
scipy.sparse.save_npz(f"{SAVE_DIR}/dosen_tfidf_matrix.npz", dosen_tfidf_matrix)

# 3. SBERT embeddings
np.save(f"{SAVE_DIR}/dosen_embeddings.npy", dosen_embeddings)

# 4. Profil dosen
dosen_profiles.to_csv(f"{SAVE_DIR}/dosen_profiles.csv", index=False)

# 5. Model SBERT
model_sbert.save(f"{SAVE_DIR}/indo_sbert_model")

# 6. Config
config = {"best_alpha": float(best_alpha)}
with open(f"{SAVE_DIR}/config.json", "w") as f:
    json.dump(config, f)
```

**Penjelasan setiap artefak yang disimpan:**

**`tfidf_vectorizer.joblib`** — TF-IDF Vectorizer yang sudah di-fit.
Format `joblib` digunakan untuk serialisasi object Python yang kompleks (seperti scikit-learn estimator). `joblib` lebih efisien dari `pickle` untuk object yang mengandung array NumPy besar karena menggunakan memory-mapped file. Vectorizer ini perlu disimpan karena mengandung kosakata (vocabulary) dan statistik IDF yang dibangun dari data training. Tanpanya, teks baru tidak dapat ditransformasi ke ruang vektor yang sama.

**`dosen_tfidf_matrix.npz`** — Matriks TF-IDF dosen dalam format sparse.
Format `npz` (NumPy compressed format) dari `scipy.sparse` digunakan karena matriks TF-IDF bersifat sparse (sebagian besar nilai adalah nol). Menyimpan sebagai sparse matrix jauh lebih hemat disk dibandingkan menyimpan sebagai dense matrix.

**`dosen_embeddings.npy`** — Embedding SBERT dosen dalam format NumPy binary.
Format `.npy` adalah format binary NumPy yang sangat efisien untuk membaca dan menulis array numerik. Embedding ini adalah matriks dense berukuran `(jumlah_dosen × 768)`.

**`dosen_profiles.csv`** — Profil teks dan metadata dosen.
Disimpan sebagai CSV yang dapat dibaca oleh manusia dan mudah diinspeksi. Berisi informasi nama dosen, keahlian, dan jumlah publikasi yang diperlukan untuk menampilkan hasil rekomendasi.

**`indo_sbert_model/`** — Model SBERT lengkap.
Menyimpan model neural network lengkap (arsitektur, bobot, dan konfigurasi tokenizer) ke sebuah direktori. Ini memungkinkan loading model di lingkungan deployment tanpa koneksi internet ke Hugging Face Hub.

**`config.json`** — Konfigurasi hyperparameter.
Menyimpan nilai `best_alpha` yang ditemukan dari grid search dalam format JSON yang mudah dibaca dan diparse di berbagai bahasa pemrograman.

**Manfaat serialisasi:**
Dalam skenario deployment (misalnya web API menggunakan Flask atau FastAPI), hanya artefak-artefak inilah yang perlu dimuat saat startup. Proses yang mahal (preprocessing semua data, training TF-IDF, encoding SBERT) hanya dilakukan sekali saat notebook ini dijalankan, bukan setiap kali request masuk.

---

## 15. Hubungan Antar Tahap dan Gambaran Pipeline Keseluruhan

### 15.1 Dependensi Antar Tahap

Setiap tahap dalam notebook ini memiliki ketergantungan yang ketat terhadap tahap sebelumnya:

```
Tahap 3 (Load Data)
    │
    └─→ Tahap 4 (Data Cleaning)
           │
           └─→ Tahap 5 (Preprocessing)
                  │
                  └─→ Tahap 6 (Profil Dosen)
                         │
                         ├─→ Tahap 7 (TF-IDF) ─────────┐
                         │                               │
                         └─→ Tahap 8 (SBERT) ──────────┤
                                                         │
                                                  Tahap 10b (Hybrid)
                                                         │
                                                  Tahap 9 (Evaluasi)
                                                         │
                                                  Tahap 10 (Perbandingan)
                                                         │
                                                  Tahap 11 (Demo)
                                                         │
                                                  Tahap 14 (Save Model)
```

### 15.2 Aliran Data

**Data skripsi** mengalir melalui dua jalur paralel:
1. **Jalur preprocessing → query**: Setiap judul skripsi dipreprocess saat digunakan sebagai query dalam `recommend_tfidf`, `recommend_sbert`, atau `recommend_hybrid`.
2. **Jalur evaluasi**: Kolom `Dosen Pembimbing` digunakan sebagai ground truth untuk mengukur akurasi.

**Data dosen** mengalir melalui satu jalur:
1. Preprocessing judul publikasi → Agregasi menjadi profil → Representasi (TF-IDF matrix atau SBERT embeddings).

### 15.3 Perbandingan Metode: Kelebihan dan Keterbatasan

| Aspek | TF-IDF (Baseline) | SBERT (Utama) | Hybrid |
|-------|-------------------|---------------|--------|
| Kemampuan semantik | Rendah — berbasis kata kunci | Tinggi — berbasis makna | Sedang–Tinggi |
| Kecepatan inferensi | Sangat cepat (operasi sparse) | Lebih lambat (neural network) | Lambat (kombinasi keduanya) |
| Kebutuhan komputasi | Minimal (CPU) | GPU direkomendasikan | GPU direkomendasikan |
| Sensitivitas bahasa | Tergantung preprocessing | Dipelajari dari data | Tergantung keduanya |
| Portabilitas | Sangat portabel | Memerlukan model besar (~500MB) | Memerlukan kedua model |
| Interpretabilitas | Tinggi (bobot fitur terlihat) | Rendah (blackbox) | Sedang |

### 15.4 Potensi Perbaikan Sistem

Beberapa area yang dapat dikembangkan lebih lanjut:

1. **Data augmentation**: Menambahkan abstrak atau kata kunci publikasi dosen (tidak hanya judul) ke profil dosen untuk representasi yang lebih kaya.

2. **Proper train/validation/test split**: Memisahkan data untuk pencarian hyperparameter (alpha) dan evaluasi final untuk menghindari optimism bias.

3. **Fine-tuning SBERT**: Melakukan fine-tuning lebih lanjut pada domain penelitian Teknik Informatika Indonesia menggunakan pasangan (judul skripsi, judul publikasi dosen pembimbing) yang relevan sebagai positive pairs.

4. **Re-ranking**: Menggunakan skor similarity sebagai kandidat awal, kemudian melakukan re-ranking berdasarkan faktor lain seperti beban bimbingan dosen aktual atau kesesuaian topik yang lebih granular.

5. **Penanganan data baru**: Sistem saat ini bersifat statis — ketika ada publikasi dosen baru, seluruh pipeline harus dijalankan ulang. Incremental update mechanism dapat meningkatkan efisiensi maintenance.

---

*Dokumentasi ini dibuat berdasarkan analisis menyeluruh terhadap notebook `RekomendasiDosenPembimbing-2.ipynb`. Seluruh penjelasan teknis mencerminkan implementasi aktual dalam notebook tersebut.*

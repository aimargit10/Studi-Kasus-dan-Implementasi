# Studi Kasus dan Implementasi

Repositori ini berisi serangkaian studi kasus dan implementasi proyek data science, machine learning, geospatial, visualisasi, dan deep learning yang dibagi ke dalam beberapa sesi.

## Daftar Sesi

### Sesi 1 - Analisis Data dan Prediksi Churn Pelanggan
- Fokus: data preprocessing, feature engineering, dan model klasifikasi churn pelanggan.
- File utama:
  - `Sesi 1/data_preprocessing.py`
  - `Sesi 1/train_model.py`
  - `Sesi 1/requirements.txt`
- Tujuan:
  - memuat dataset churn,
  - membersihkan data,
  - normalisasi numerik,
  - encoding kategorikal,
  - membandingkan Logistic Regression dan Random Forest.

### Sesi 2 - Integrasi Data Lingkungan / Cuaca
- Fokus: pipeline ETL untuk data cuaca dari API publik.
- File utama:
  - `Sesi 2/fetcher.py`
  - `Sesi 2/processor.py`
  - `Sesi 2/main.py`
  - `Sesi 2/requirements.txt`
- Tujuan:
  - mengambil data dari Open-Meteo,
  - membersihkan JSON mentah,
  - menyusun DataFrame Pandas,
  - menyimpan hasil ke CSV lokal.

### Sesi 3 - Geospatial Data Engineering
- Fokus: analisis data spasial dan visualisasi peta.
- File utama:
  - `Sesi 3/geospatial_analysis.py`
  - `Sesi 3/requirements.txt`
- Tujuan:
  - memuat data GeoJSON,
  - memfilter data berdasarkan atribut,
  - membuat peta interaktif dengan Folium,
  - membuat peta statis dengan Matplotlib.

### Sesi 4 - Machine Learning Pipeline
- Fokus: pemodelan klasifikasi menggunakan Scikit-Learn.
- File utama:
  - `Sesi 4/ml_pipeline.py`
  - `Sesi 4/requirements.txt`
- Tujuan:
  - memuat dataset Breast Cancer dari sklearn,
  - split data train-test,
  - scale fitur dengan StandardScaler,
  - melatih Decision Tree dan SVM,
  - mengevaluasi training vs testing dan classification report.

### Sesi 5 - Dashboard Interaktif dengan Streamlit
- Fokus: visualisasi data interaktif yang bersih dan informatif.
- File utama:
  - `Sesi 5/app.py`
  - `Sesi 5/requirements.txt`
- Tujuan:
  - sidebar filter interaktif,
  - overview chart utama,
  - tabel detail data,
  - menerapkan prinsip overview first dan details on demand.

### Sesi 6 - Deep Learning dengan PyTorch
- Fokus: klasifikasi citra sederhana menggunakan CNN.
- File utama:
  - `Sesi 6/cnn_model.py`
  - `Sesi 6/requirements.txt`
- Tujuan:
  - membangun CNN sederhana,
  - melatih model dengan Adam dan CrossEntropyLoss,
  - menghitung akurasi pada data uji.

---

## Struktur Folder

```text
Studi Kasus dan Implementasi/
├── README.md
├── Sesi 1/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   └── requirements.txt
├── Sesi 2/
│   ├── fetcher.py
│   ├── processor.py
│   ├── main.py
│   └── requirements.txt
├── Sesi 3/
│   ├── geospatial_analysis.py
│   └── requirements.txt
├── Sesi 4/
│   ├── ml_pipeline.py
│   └── requirements.txt
├── Sesi 5/
│   ├── app.py
│   └── requirements.txt
├── Sesi 6/
│   ├── cnn_model.py
│   └── requirements.txt
└── ...
```

---

## Persiapan Umum

Pastikan Python sudah terinstal di sistem Anda. Disarankan menggunakan virtual environment agar dependensi lebih terkelola.

```bash
python -m venv .venv
.venv\Scripts\activate
```

Pada Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

---

## Cara Menjalankan Setiap Sesi

### Sesi 1
```bash
cd "Sesi 1"
pip install -r requirements.txt
python train_model.py
```

### Sesi 2
```bash
cd "Sesi 2"
pip install -r requirements.txt
python main.py
```

### Sesi 3
```bash
cd "Sesi 3"
pip install -r requirements.txt
python geospatial_analysis.py
```

### Sesi 4
```bash
cd "Sesi 4"
pip install -r requirements.txt
python ml_pipeline.py
```

### Sesi 5
```bash
cd "Sesi 5"
pip install -r requirements.txt
streamlit run app.py
```

### Sesi 6
```bash
cd "Sesi 6"
pip install -r requirements.txt
python cnn_model.py
```

---

## Catatan

- Beberapa sesi menggunakan dataset dummy atau API publik gratis agar proses berjalan tanpa kebutuhan dataset lokal yang kompleks.
- Pastikan koneksi internet aktif untuk sesi yang memanggil API publik.
- Untuk Sesi 5, jalankan aplikasi menggunakan Streamlit agar dashboard dapat diakses di browser.

---

## Tujuan Proyek

Proyek ini dibuat untuk memberikan gambaran komprehensif tentang pipeline kerja data science dari tahap awal hingga deployment sederhana, mencakup:
- preprocessing data,
- machine learning,
- visualisasi interaktif,
- geospatial analysis,
- deep learning,
- dan integrasi data secara umum.

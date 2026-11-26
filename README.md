# ✈️ Estimasi Harga Tiket Pesawat

Team-Based Project mata kuliah **Data Mining** — Informatika, Universitas Sebelas Maret.

Proyek ini menebak harga tiket pesawat dari ciri-ciri penerbangannya (maskapai, kelas, rute, jumlah transit, durasi, dan kapan tiket dibeli), membandingkan tiga algoritma regresi, lalu mencari faktor yang paling menentukan harga.

## Rumusan masalah
1. Bisakah harga tiket ditebak dari ciri-ciri penerbangannya?
2. Algoritma mana yang tebakannya paling akurat?
3. Faktor apa yang paling menentukan harga?
4. Apa saran untuk penumpang dan maskapai?

## Hasil singkat

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---|---|---|
| **Decision Tree (max_depth=25)** | **1.218** | **2.998** | **0,983** |
| KNN Regression (k=3) | 1.786 | 3.856 | 0,971 |
| Multiple Linear Regression per kelas | 3.210 | 5.673 | 0,938 |
| Multiple Linear Regression gabungan | 4.935 | 9.054 | 0,841 |

Diuji pada 60.031 tiket (20% data) yang tidak dipakai saat pelatihan.

Faktor paling berpengaruh: **kelas** (Business ≈ 7,7× Economy), lalu **waktu pemesanan** (Economy yang dibeli 1–3 hari sebelum berangkat ±2,7× lebih mahal daripada ≥16 hari), **transit/durasi**, dan **maskapai**.

## Struktur folder

```
flight-price-estimation/
├── configs/          # parameter preprocessing & model (config.yaml)
├── data/
│   ├── raw/          # Clean_Dataset.csv dari Kaggle (tidak di-commit)
│   └── processed/    # train.csv & test.csv hasil preprocessing (tidak di-commit)
├── docs/             # laporan, slide presentasi, dokumen pendukung
├── notebooks/        # 01_EDA → 02_Preprocessing → 03_Modeling_Evaluasi
├── outputs/
│   ├── figures/      # grafik EDA & evaluasi
│   ├── models/       # model_dt.joblib & fitur_dt.json untuk dashboard
│   └── results/      # tabel hasil (perbandingan model, koefisien MLR)
├── scripts/          # app.py — dashboard Streamlit
├── requirements.txt
└── README.md
```

## Dataset
[Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) (Kaggle, Shubham Bathwal): ±300 ribu data tiket penerbangan domestik antar 6 kota besar di India, harga dalam rupee (₹).

Dataset tidak disimpan di repo. Unduh dari Kaggle, lalu letakkan `Clean_Dataset.csv` di `data/raw/`.

## Cara menjalankan

### Notebook
Notebook dijalankan di Google Colab dengan data di Google Drive tim, berurutan:
`01_EDA` → `02_Preprocessing` → `03_Modeling_Evaluasi`.
Sesuaikan variabel path (`BASE` / `ROOT`) di sel pertama setiap notebook jika lokasi data berbeda.

### Dashboard
```bash
uv venv --python 3.12          # atau: python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt   # atau: pip install -r requirements.txt
streamlit run scripts/app.py
```
Dashboard terbuka di `http://localhost:8501`.

## Anggota kelompok
| Nama | NIM | Peran |
|---|---|---|
| [Nama] | [NIM] | [Peran] |
| [Nama] | [NIM] | [Peran] |
| [Nama] | [NIM] | [Peran] |
| [Nama] | [NIM] | [Peran] |
| [Nama] | [NIM] | [Peran] |

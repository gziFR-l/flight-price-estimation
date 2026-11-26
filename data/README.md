# Data

Folder ini berisi dataset asli dan hasil preprocessing yang dipakai di proyek.

```
data/
├── raw/          # dataset asli dari Kaggle
│   └── Clean_Dataset.csv
└── processed/    # hasil notebook 02_Preprocessing
    ├── train.csv
    └── test.csv
```

## Sumber dataset

**Flight Price Prediction** — Shubham Bathwal, Kaggle
https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

Berisi data tiket penerbangan domestik antar 6 kota besar di India. Harga dalam rupee India (₹).

| Keterangan | Nilai |
|---|---|
| Jumlah baris | 300.153 tiket |
| Jumlah kolom | 12 (termasuk kolom indeks `Unnamed: 0`) |
| Missing value | Tidak ada |
| Duplikat | Tidak ada |
| Kelas | Economy 206.666 (69%) · Business 93.487 (31%) |
| Rentang harga | ₹1.105 – ₹123.071 |

## Asal file

`raw/Clean_Dataset.csv` diunduh dari Kaggle (link di atas). Paket unduhan Kaggle juga berisi `business.csv` dan `economy.csv`, yaitu data mentah sebelum dibersihkan; keduanya tidak dipakai di proyek ini.

`processed/train.csv` dan `processed/test.csv` bisa dibuat ulang kapan saja dengan menjalankan notebook `02_Preprocessing`.

> Notebook di repo ini dijalankan di Google Colab dengan data di Google Drive tim. Jika menjalankan di tempat lain, sesuaikan variabel `BASE` / `ROOT` di sel pertama setiap notebook.

## Kolom pada `raw/Clean_Dataset.csv`

| Kolom | Keterangan | Contoh nilai |
|---|---|---|
| `Unnamed: 0` | Nomor indeks (dibuang) | 0, 1, 2 |
| `airline` | Maskapai | AirAsia, Air_India, GO_FIRST, Indigo, SpiceJet, Vistara |
| `flight` | Kode penerbangan (dibuang) | SG-8709, UK-995 |
| `source_city` | Kota asal | Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai |
| `departure_time` | Slot waktu berangkat | Early_Morning, Morning, Afternoon, Evening, Night, Late_Night |
| `stops` | Jumlah transit | zero, one, two_or_more |
| `arrival_time` | Slot waktu tiba | sama dengan `departure_time` |
| `destination_city` | Kota tujuan | sama dengan `source_city` |
| `class` | Kelas penerbangan | Economy, Business |
| `duration` | Lama penerbangan dalam jam desimal | 2.17 (±2 jam 10 menit) |
| `days_left` | Sisa hari antara pemesanan dan keberangkatan | 1 – 49 |
| `price` | Harga tiket (₹) — **target** | 5953 |

## Isi `processed/` (hasil `02_Preprocessing`)

| File | Baris | Kolom |
|---|---|---|
| `train.csv` | 240.122 (80%) | 34 |
| `test.csv` | 60.031 (20%) | 34 |

Pembagian dilakukan secara acak dengan `random_state=42`.

Perubahan dari data asli:

| Perubahan | Hasil |
|---|---|
| Kolom `Unnamed: 0` dan `flight` dibuang | — |
| `stops` diubah ke angka | 0 / 1 / 2 |
| `class` diubah menjadi `is_business` | 0 = Economy, 1 = Business |
| Fitur baru `days_group` dari `days_left` | 1-3, 4-10, 11-15, 16+ hari |
| One-hot encoding (`drop_first=True`) | `airline_*`, `source_city_*`, `destination_city_*`, `departure_time_*`, `arrival_time_*`, `days_group_*` |
| Target baru `log_price` | log natural dari `price` |

Kategori baseline (kolom yang dibuang saat one-hot encoding): maskapai **AirAsia**, kota asal dan tujuan **Bangalore**, waktu berangkat dan tiba **Afternoon**, kelompok pemesanan **16+ hari**.

Kolom `days_left` asli tetap disimpan untuk Decision Tree dan KNN; `days_group_*` dipakai oleh Multiple Linear Regression. Parameter lengkapnya ada di [`configs/config.yaml`](../configs/config.yaml).

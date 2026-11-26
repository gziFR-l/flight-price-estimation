"""
Dashboard Estimasi Harga Tiket Pesawat
TBP Data Mining — model Decision Tree Regression (target: log harga)

Jalankan:  streamlit run app.py
File yang harus ada di folder yang sama: model_dt.joblib, fitur_dt.json
(opsional: perbandingan_model.csv, pengaruh_per_kelas_mlr.csv dari folder Evaluasi)
"""
import json
import os
from datetime import date, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Estimasi Harga Tiket Pesawat", page_icon="✈️", layout="wide")


# ---------- memuat model ----------
@st.cache_resource
def muat_model():
    model = joblib.load(os.path.join(HERE, "model_dt.joblib"))
    with open(os.path.join(HERE, "fitur_dt.json")) as f:
        fitur = json.load(f)
    return model, fitur


model, FITUR = muat_model()

MASKAPAI = ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
PUNYA_BUSINESS = {"Air_India", "Vistara"}
KOTA = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
SLOT = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
LABEL_SLOT = {
    "Early_Morning": "Dini pagi", "Morning": "Pagi", "Afternoon": "Siang",
    "Evening": "Sore", "Night": "Malam", "Late_Night": "Larut malam",
}
LABEL_STOPS = {0: "Langsung (tanpa transit)", 1: "1 transit", 2: "2 transit atau lebih"}
DURASI_DEFAULT = {0: 2.2, 1: 11.0, 2: 16.0}  # perkiraan awal, bisa diubah pengguna


def susun_fitur(maskapai, asal, tujuan, berangkat, tiba, stops, kelas, durasi, days_left):
    """Menyusun satu baris fitur dengan urutan kolom yang sama seperti saat training."""
    baris = dict.fromkeys(FITUR, 0)
    baris.update({"stops": stops, "duration": durasi, "days_left": days_left,
                  "is_business": int(kelas == "Business")})
    for nama in (f"airline_{maskapai}", f"source_city_{asal}", f"destination_city_{tujuan}",
                 f"departure_time_{berangkat}", f"arrival_time_{tiba}"):
        if nama in baris:  # kategori baseline tidak punya kolom dummy → tetap 0
            baris[nama] = 1
    return pd.DataFrame([baris])[FITUR]


def prediksi(df_fitur):
    return np.exp(model.predict(df_fitur))


def rupee(x):
    return f"₹{x:,.0f}".replace(",", ".")


# ---------- sidebar: input ----------
st.sidebar.header("Detail Penerbangan")
maskapai = st.sidebar.selectbox("Maskapai", MASKAPAI, index=MASKAPAI.index("Vistara"))
pilihan_kelas = ["Economy", "Business"] if maskapai in PUNYA_BUSINESS else ["Economy"]
kelas = st.sidebar.radio("Kelas", pilihan_kelas, horizontal=True)
if maskapai not in PUNYA_BUSINESS:
    st.sidebar.caption("Di dataset, kelas Business hanya tersedia di Vistara dan Air India.")

asal = st.sidebar.selectbox("Kota asal", KOTA, index=KOTA.index("Delhi"))
tujuan = st.sidebar.selectbox("Kota tujuan", [k for k in KOTA if k != asal])

stops = st.sidebar.select_slider("Jumlah transit", options=[0, 1, 2], value=1,
                                 format_func=lambda s: LABEL_STOPS[s])
durasi = st.sidebar.number_input("Durasi penerbangan (jam)", min_value=0.8, max_value=50.0,
                                 value=DURASI_DEFAULT[stops], step=0.25)
berangkat = st.sidebar.selectbox("Waktu berangkat", SLOT, index=1, format_func=LABEL_SLOT.get)
tiba = st.sidebar.selectbox("Waktu tiba", SLOT, index=3, format_func=LABEL_SLOT.get)

hari_ini = date.today()
tgl = st.sidebar.date_input("Tanggal berangkat", value=hari_ini + timedelta(days=20),
                            min_value=hari_ini + timedelta(days=1),
                            max_value=hari_ini + timedelta(days=49))
days_left = (tgl - hari_ini).days
st.sidebar.caption(f"Dipesan **{days_left} hari** sebelum keberangkatan "
                   "(rentang data training: 1–49 hari).")

# ---------- halaman utama ----------
st.title("✈️ Estimasi Harga Tiket Pesawat")
st.caption("Prototype TBP Data Mining · model Decision Tree Regression · "
           "data penerbangan domestik India (Kaggle: Flight Price Prediction)")

tab1, tab2 = st.tabs(["Estimasi Harga", "Tentang Model"])

with tab1:
    x = susun_fitur(maskapai, asal, tujuan, berangkat, tiba, stops, kelas, durasi, days_left)
    harga = prediksi(x)[0]

    # harga untuk setiap kemungkinan waktu pemesanan (1–49 hari) dengan detail penerbangan yang sama
    semua_hari = pd.concat(
        [susun_fitur(maskapai, asal, tujuan, berangkat, tiba, stops, kelas, durasi, d) for d in range(1, 50)],
        ignore_index=True)
    kurva = pd.DataFrame({"Sisa hari sebelum berangkat": range(1, 50), "Estimasi harga (INR)": prediksi(semua_hari)})
    termurah = kurva.loc[kurva["Estimasi harga (INR)"].idxmin()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimasi harga", rupee(harga))
    c2.metric("Agar dapat harga termurah, pesan paling lambat",
              f"{int(termurah['Sisa hari sebelum berangkat'])} hari sebelum berangkat",
              rupee(termurah["Estimasi harga (INR)"]), delta_color="off")
    selisih = (harga / termurah["Estimasi harga (INR)"] - 1) * 100
    c3.metric("Lebih mahal dari harga termurah", f"{selisih:.0f}%")

    st.subheader("Estimasi harga menurut waktu pemesanan")
    st.line_chart(kurva, x="Sisa hari sebelum berangkat", y="Estimasi harga (INR)")
    st.caption("Garis menunjukkan estimasi harga untuk penerbangan yang sama jika dipesan "
               "1–49 hari sebelum keberangkatan. Semakin ke kiri, semakin mendadak.")

    if days_left <= 3:
        st.warning("Pemesanan sangat mendadak (≤3 hari). Di data, ini kelompok dengan harga tertinggi.")
    elif days_left >= 16 and kelas == "Economy":
        st.success("Waktu pemesanan sudah ideal: untuk Economy, harga relatif stabil jika dipesan ≥16 hari sebelumnya.")

with tab2:
    st.subheader("Perbandingan model")
    path_tabel = os.path.join(HERE, "perbandingan_model.csv")
    if os.path.exists(path_tabel):
        st.dataframe(pd.read_csv(path_tabel, index_col=0), width="stretch")
    else:
        st.info("Salin `perbandingan_model.csv` dari folder Evaluasi ke folder ini untuk menampilkan tabel.")

    st.subheader("Pengaruh tiap faktor terhadap harga (MLR per kelas)")
    path_efek = os.path.join(HERE, "pengaruh_per_kelas_mlr.csv")
    if os.path.exists(path_efek):
        efek = pd.read_csv(path_efek, index_col=0)
        st.dataframe(efek.style.format("{:+.1f}%", na_rep="–"), width="stretch")
        st.caption("Persentase perubahan harga dibanding kategori baseline: maskapai AirAsia (Economy) / "
                   "Air India (Business), pemesanan ≥16 hari, per tambahan 1 transit, per tambahan 1 jam durasi.")
    else:
        st.info("Salin `pengaruh_per_kelas_mlr.csv` dari folder Evaluasi ke folder ini untuk menampilkan tabel.")

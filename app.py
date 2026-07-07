import streamlit as st
import pandas as pd
import pickle

# 1. LOAD MODEL
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# 2. TAMPILAN UTAMA APLIKASI
st.title("💳 Credit Card Default Prediction App")
st.write("Aplikasi ini memprediksi apakah nasabah akan gagal bayar bulan depan.")

st.sidebar.header("Input Data Nasabah")

# 3. FORM INPUT (Sesuai dengan fitur yang digunakan saat training)
limit_bal = st.sidebar.number_input("Limit Kredit (LIMIT_BAL)", min_value=0, value=50000)
sex = st.sidebar.selectbox("Jenis Kelamin (SEX)", options=[1, 2], format_func=lambda x: "Pria" if x==1 else "Wanita")
education = st.sidebar.selectbox("Pendidikan (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: ["Graduate School", "University", "High School", "Others"][x-1])
marriage = st.sidebar.selectbox("Status Pernikahan (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: ["Menikah", "Lajang", "Lainnya"][x-1])
age = st.sidebar.slider("Usia (AGE)", 18, 100, 30)

# Masukkan status pembayaran bulan terakhir (PAY_0) sebagai contoh pemicu utama
pay_0 = st.sidebar.slider("Status Pembayaran Bulan Terakhir (PAY_0)", -1, 9, 0)

# (Catatan: Tambahkan input untuk fitur BILL_AMT dan PAY_AMT lainnya sesuai kebutuhan model Anda)

# 4. LOGIKA PREDIKSI
if st.button("Prediksi Sekarang"):
    # Buat dataframe dari input data baru (pastikan susunan kolom SAMA dengan saat training)
    # Sebagai contoh ringkas, isi kolom lainnya dengan nilai default jika tidak semua ditampilkan di UI
    data_baru = pd.DataFrame({
        'LIMIT_BAL': [limit_bal], 'SEX': [sex], 'EDUCATION': [education], 'MARRIAGE': [marriage], 'AGE': [age],
        'PAY_0': [pay_0], 'PAY_2': [0], 'PAY_3': [0], 'PAY_4': [0], 'PAY_5': [0], 'PAY_6': [0],
        'BILL_AMT1': [0], 'BILL_AMT2': [0], 'BILL_AMT3': [0], 'BILL_AMT4': [0], 'BILL_AMT5': [0], 'BILL_AMT6': [0],
        'PAY_AMT1': [0], 'PAY_AMT2': [0], 'PAY_AMT3': [0], 'PAY_AMT4': [0], 'PAY_AMT5': [0], 'PAY_AMT6': [0]
    })
    
    # Lakukan prediksi
    prediksi = model.predict(data_baru)[0]
    
    # Tampilkan Hasil
    st.subheader("Hasil Analisis:")
    if prediksi == 1:
        st.error("⚠️ Nasabah berpotensi GAGAL BAYAR (Default) bulan depan.")
    else:
        st.success("✅ Nasabah aman / TIDAK DEFAULT.")
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier # Anda bisa ganti dengan XGBClassifier atau CatBoostClassifier jika itu model terbaik Anda

# Fungsi untuk membuat dan melatih model secara otomatis di server
@st.cache_resource
def build_and_train_model():
    # 1. Ambil data langsung dari file CSV di repositori Anda
    df = pd.read_csv('UCI_Credit_Card.csv')
    
    # Pembersihan kolom bawaan jika ada
    if 'ID' in df.columns: 
        df = df.drop(columns=['ID'])
    if 'index' in df.columns: 
        df = df.drop(columns=['index'])
        
    # Standardisasi kategori sesuai notebook Anda
    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})
    
    # Deteksi nama kolom target
    target_col = 'default.payment.next.month' if 'default.payment.next.month' in df.columns else 'default'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Definisikan Pipeline Preprocessing yang sama dengan isi notebook
    numeric_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ])
    
    # 3. Satukan Preprocessor dengan Model Algoritma Terbaik Anda
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42))
    ])
    
    # Jalankan training model langsung di server Streamlit
    model_pipeline.fit(X, y)
    return model_pipeline

# Panggil fungsi training otomatis
model = build_and_train_model()

# ==========================================
# TAMPILAN ANTARMUKA WEB STREAMLIT
# ==========================================
st.set_page_config(page_title="Credit Card Default Prediction", page_icon="💳")
st.title("💳 Credit Card Default Prediction App")
st.write("Aplikasi Prediksi Potensi Gagal Bayar Nasabah Bulan Depan.")

st.sidebar.header("Input Parameter Nasabah")

# Pembuatan komponen Form Input di Sidebar kiri
limit_bal = st.sidebar.number_input("Limit Kredit (LIMIT_BAL)", min_value=0, value=50000)
sex = st.sidebar.selectbox("Jenis Kelamin (SEX)", options=[1, 2], format_func=lambda x: "Pria" if x==1 else "Wanita")
education = st.sidebar.selectbox("Pendidikan (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: ["Graduate School", "University", "High School", "Others"][x-1])
marriage = st.sidebar.selectbox("Status Pernikahan (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: ["Menikah", "Lajang", "Lainnya"][x-1])
age = st.sidebar.slider("Usia (AGE)", 18, 100, 30)
pay_0 = st.sidebar.slider("Status Pembayaran Bulan Terakhir (PAY_0)", -2, 8, 0)

# Tombol Eksekusi Prediksi
if st.button("Prediksi Kemungkinan Gagal Bayar"):
    # Menyusun data baru ke bentuk DataFrame
    data_baru = pd.DataFrame({
        'LIMIT_BAL': [limit_bal], 'SEX': [sex], 'EDUCATION': [education], 'MARRIAGE': [marriage], 'AGE': [age],
        'PAY_0': [pay_0], 'PAY_2': [0], 'PAY_3': [0], 'PAY_4': [0], 'PAY_5': [0], 'PAY_6': [0],
        'BILL_AMT1': [0], 'BILL_AMT2': [0], 'BILL_AMT3': [0], 'BILL_AMT4': [0], 'BILL_AMT5': [0], 'BILL_AMT6': [0],
        'PAY_AMT1': [0], 'PAY_AMT2': [0], 'PAY_AMT3': [0], 'PAY_AMT4': [0], 'PAY_AMT5': [0], 'PAY_AMT6': [0]
    })
    
    # Prediksi menggunakan pipeline yang sudah di-training otomatis tadi
    prediksi = model.predict(data_baru)[0]
    
    st.subheader("Hasil Analisis Real-Time:")
    if prediksi == 1:
        st.error("⚠️ Peringatan: Nasabah berpotensi GAGAL BAYAR (*Default*) bulan depan!")
    else:
        st.success("✅ Aman: Nasabah diprediksi akan membayar tagihan dengan lancar (*Tidak Default*).")

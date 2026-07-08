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
st.set_page_config(page_title="Credit Card Default Prediction", page_icon=None, layout="centered")

# ------------------------------------------
# CUSTOM CSS - TEMA PINK
# ------------------------------------------
st.markdown("""
<style>
/* Background utama */
.stApp {
    background: linear-gradient(180deg, #fff0f6 0%, #ffe3ee 100%);
}

/* Judul utama */
.title-box {
    background: linear-gradient(135deg, #ff6fa5, #ff9ec4);
    padding: 28px 24px;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(255, 111, 165, 0.35);
    margin-bottom: 20px;
    text-align: center;
}
.title-box h1 {
    color: white;
    font-size: 30px;
    margin-bottom: 6px;
    font-weight: 800;
    letter-spacing: 0.5px;
}
.title-box p {
    color: #fff0f6;
    font-size: 15px;
    margin: 0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffdcec 0%, #ffc9e1 100%);
    border-right: 3px solid #ff6fa5;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] label {
    color: #c2185b !important;
    font-weight: 700 !important;
}

/* Tombol prediksi */
div.stButton > button {
    background: linear-gradient(135deg, #ff4d94, #ff85b3);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 30px;
    padding: 12px 26px;
    width: 100%;
    box-shadow: 0 6px 14px rgba(255, 77, 148, 0.4);
    transition: 0.25s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 10px 20px rgba(255, 77, 148, 0.5);
}

/* Subheader hasil */
.result-header {
    color: #c2185b;
    font-weight: 800;
    font-size: 22px;
    margin: 18px 0 10px 0;
    border-left: 6px solid #ff6fa5;
    padding-left: 10px;
}

/* Kartu hasil */
.result-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 20px;
    border-radius: 16px;
    margin-top: 8px;
    font-size: 16px;
    font-weight: 600;
}
.result-card.danger {
    background: #fff0f4;
    border: 2px solid #ff4d6d;
    color: #b3123c;
}
.result-card.safe {
    background: #f1fff6;
    border: 2px solid #33c481;
    color: #1a7a4d;
}

/* Ikon segitiga peringatan */
.icon-warning {
    width: 0;
    height: 0;
    border-left: 16px solid transparent;
    border-right: 16px solid transparent;
    border-bottom: 28px solid #ff4d6d;
    position: relative;
    flex-shrink: 0;
}
.icon-warning::after {
    content: "!";
    position: absolute;
    top: 10px;
    left: -4px;
    color: white;
    font-weight: 900;
    font-size: 13px;
}

/* Ikon centang (checkmark) dari border */
.icon-check {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: #33c481;
    position: relative;
    flex-shrink: 0;
}
.icon-check::after {
    content: "";
    position: absolute;
    left: 9px;
    top: 5px;
    width: 7px;
    height: 12px;
    border: solid white;
    border-width: 0 3px 3px 0;
    transform: rotate(45deg);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# HEADER
# ------------------------------------------
st.markdown("""
<div class="title-box">
    <h1>Credit Card Default Prediction</h1>
    <p>Aplikasi Prediksi Potensi Gagal Bayar Nasabah Bulan Depan</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Input Parameter Nasabah")

# Pembuatan komponen Form Input di Sidebar kiri
limit_bal = st.sidebar.number_input("Limit Kredit (LIMIT_BAL)", min_value=0, value=50000)
sex = st.sidebar.selectbox("Jenis Kelamin (SEX)", options=[1, 2], format_func=lambda x: "Pria" if x == 1 else "Wanita")
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

    st.markdown('<div class="result-header">Hasil Analisis Real-Time</div>', unsafe_allow_html=True)

    if prediksi == 1:
        st.markdown("""
        <div class="result-card danger">
            <div class="icon-warning"></div>
            <div>Peringatan: Nasabah berpotensi <b>GAGAL BAYAR (Default)</b> bulan depan!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card safe">
            <div class="icon-check"></div>
            <div>Aman: Nasabah diprediksi akan membayar tagihan dengan lancar <b>(Tidak Default)</b>.</div>
        </div>
        """, unsafe_allow_html=True)

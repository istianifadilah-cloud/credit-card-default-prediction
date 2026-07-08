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
# CUSTOM CSS - TEMA PASTEL SENJA YANG SOFT
# ------------------------------------------
st.markdown("""
<style>
/* Background utama - gradasi senja pastel: lavender - pink - peach */
.stApp {
    background: linear-gradient(160deg, #f4e4f7 0%, #fbdce7 28%, #ffe0e0 52%, #ffe7d1 76%, #fff2dc 100%);
}

/* Header + karakter maskot */
.title-box {
    position: relative;
    background: linear-gradient(135deg, #f4a6c9 0%, #f7b6a3 100%);
    padding: 30px 24px 26px 24px;
    border-radius: 22px;
    box-shadow: 0 10px 26px rgba(244, 166, 201, 0.35);
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.title-text h1 {
    color: #fffaf6;
    font-size: 27px;
    margin-bottom: 6px;
    font-weight: 800;
    letter-spacing: 0.3px;
    text-shadow: 0 2px 6px rgba(150, 70, 100, 0.15);
}
.title-text p {
    color: #fff4ee;
    font-size: 14px;
    margin: 0;
}

/* ===== KARAKTER MASKOT ROBOT (murni CSS) ===== */
.mascot {
    position: relative;
    width: 74px;
    height: 84px;
    flex-shrink: 0;
}
.mascot .antena {
    position: absolute;
    top: -14px;
    left: 33px;
    width: 4px;
    height: 14px;
    background: #fff4ee;
    border-radius: 2px;
}
.mascot .antena::after {
    content: "";
    position: absolute;
    top: -8px;
    left: -3px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #ffe5a3;
    box-shadow: 0 0 8px rgba(255, 229, 163, 0.8);
}
.mascot .kepala {
    position: absolute;
    top: 4px;
    left: 7px;
    width: 60px;
    height: 50px;
    background: #fffaf6;
    border-radius: 22px;
    box-shadow: inset 0 -6px 0 rgba(244, 166, 201, 0.2);
}
.mascot .mata {
    position: absolute;
    top: 22px;
    width: 8px;
    height: 11px;
    background: #a9718c;
    border-radius: 50%;
}
.mascot .mata.kiri { left: 18px; }
.mascot .mata.kanan { left: 40px; }
.mascot .pipi {
    position: absolute;
    top: 32px;
    width: 8px;
    height: 5px;
    background: #f8c1d4;
    border-radius: 50%;
    opacity: 0.9;
}
.mascot .pipi.kiri { left: 10px; }
.mascot .pipi.kanan { left: 52px; }
.mascot .senyum {
    position: absolute;
    top: 36px;
    left: 27px;
    width: 14px;
    height: 7px;
    border-bottom: 3px solid #a9718c;
    border-radius: 0 0 12px 12px;
}
.mascot .badan {
    position: absolute;
    top: 50px;
    left: 15px;
    width: 44px;
    height: 30px;
    background: #fff0e8;
    border-radius: 16px 16px 20px 20px;
    box-shadow: inset 0 -5px 0 rgba(244, 166, 201, 0.18);
}
.mascot .panel {
    position: absolute;
    top: 58px;
    left: 30px;
    width: 14px;
    height: 14px;
    background: linear-gradient(135deg, #f4a6c9, #f7b6a3);
    border-radius: 50%;
}
.mascot .lengan {
    position: absolute;
    top: 54px;
    width: 8px;
    height: 16px;
    background: #fffaf6;
    border-radius: 6px;
}
.mascot .lengan.kiri { left: 2px; transform: rotate(-15deg); }
.mascot .lengan.kanan { left: 64px; transform: rotate(15deg); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffe9d6 0%, #ffdcd1 45%, #f6c9d9 100%);
    border-right: 3px solid #f4a6c9;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] label {
    color: #a9527a !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stSelectbox {
    background: rgba(255, 255, 255, 0.4);
    border-radius: 12px;
    padding: 6px 8px;
}

/* Tombol prediksi */
div.stButton > button {
    background: linear-gradient(135deg, #f4a6c9, #f7b6a3);
    color: #fffaf6;
    font-weight: 700;
    border: none;
    border-radius: 30px;
    padding: 12px 26px;
    width: 100%;
    box-shadow: 0 6px 18px rgba(244, 166, 201, 0.4);
    transition: 0.25s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 10px 24px rgba(244, 166, 201, 0.5);
    filter: brightness(1.04);
}

/* Subheader hasil */
.result-header {
    color: #a9527a;
    font-weight: 800;
    font-size: 22px;
    margin: 18px 0 10px 0;
    border-left: 6px solid #f4a6c9;
    padding-left: 10px;
}

/* Kartu hasil */
.result-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 20px;
    border-radius: 18px;
    margin-top: 8px;
    font-size: 16px;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}
.result-card.danger {
    background: #fff1ee;
    border: 2px solid #f4a3a3;
    color: #b05a5a;
}
.result-card.safe {
    background: #fff8ec;
    border: 2px solid #f3c98d;
    color: #b1793a;
}

/* Ikon wajah kecil murni CSS untuk hasil (bukan emoji) */
.face-icon {
    position: relative;
    width: 34px;
    height: 34px;
    border-radius: 50%;
    flex-shrink: 0;
}
.face-icon.worried { background: #ffe3df; border: 2px solid #f4a3a3; }
.face-icon.happy { background: #fff2d9; border: 2px solid #f3c98d; }

.face-icon .eye {
    position: absolute;
    top: 12px;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: #8f5555;
}
.face-icon.happy .eye { background: #a1743a; }
.face-icon .eye.l { left: 9px; }
.face-icon .eye.r { left: 20px; }

.face-icon.worried .mouth {
    position: absolute;
    top: 20px;
    left: 10px;
    width: 14px;
    height: 6px;
    border-top: 2.5px solid #8f5555;
    border-radius: 12px 12px 0 0;
}
.face-icon.happy .mouth {
    position: absolute;
    top: 18px;
    left: 10px;
    width: 14px;
    height: 7px;
    border-bottom: 2.5px solid #a1743a;
    border-radius: 0 0 12px 12px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------
# HEADER + MASKOT
# ------------------------------------------
st.markdown("""
<div class="title-box">
    <div class="title-text">
        <h1>Credit Card Default Prediction</h1>
        <p>Aplikasi Prediksi Potensi Gagal Bayar Nasabah Bulan Depan</p>
    </div>
    <div class="mascot">
        <div class="antena"></div>
        <div class="kepala"></div>
        <div class="mata kiri"></div>
        <div class="mata kanan"></div>
        <div class="pipi kiri"></div>
        <div class="pipi kanan"></div>
        <div class="senyum"></div>
        <div class="lengan kiri"></div>
        <div class="lengan kanan"></div>
        <div class="badan"></div>
        <div class="panel"></div>
    </div>
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
            <div class="face-icon worried">
                <div class="eye l"></div>
                <div class="eye r"></div>
                <div class="mouth"></div>
            </div>
            <div>Peringatan: Nasabah berpotensi <b>GAGAL BAYAR (Default)</b> bulan depan!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card safe">
            <div class="face-icon happy">
                <div class="eye l"></div>
                <div class="eye r"></div>
                <div class="mouth"></div>
            </div>
            <div>Aman: Nasabah diprediksi akan membayar tagihan dengan lancar <b>(Tidak Default)</b>.</div>
        </div>
        """, unsafe_allow_html=True)

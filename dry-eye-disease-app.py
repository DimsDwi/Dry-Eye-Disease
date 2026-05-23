import streamlit as st
import numpy as np
import joblib
import time
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA
# ==========================================
st.set_page_config(
    page_title="Dry Eye Clinical Analytics", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="collapsed" # Sidebar disembunyikan di awal
)

# ==========================================
# 2. CUSTOM CSS (Premium Soft Dark Medical Mode)
# ==========================================
# Catatan: Bagian 'header' tidak disembunyikan agar tombol DEPLOY tetap ada!
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Mengatur Font Utama dan Background Gelap Lembut */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background-color: #111827 !important; /* Warna Latar Abu-abu Arang Matte */
        color: #F3F4F6 !important; /* Warna Teks Putih Halus */
    }
    
    /* Styling Judul Utama - Cyan Medical */
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #06B6D4; /* Medical Cyan */
        margin-bottom: 0px;
        letter-spacing: -1.5px;
    }
    .main-subtitle {
        font-size: 16px;
        color: #9CA3AF;
        margin-top: 5px;
        margin-bottom: 30px;
        font-weight: 400;
    }

    /* Kustomisasi Container agar seperti Kartu Melayang (Dark Cards) */
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background-color: #1F2937 !important; /* Slightly lighter gray-dark */
        border: 1px solid #374151 !important; /* Subtle border */
        border-radius: 20px !important;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2) !important; /* Softer dark shadow */
        padding: 10px !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="stVerticalBlock"] > div[style*="border"]:hover {
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3) !important;
        transform: translateY(-2px);
    }

    /* Merubah Tombol Prediksi Menjadi Elegan & Menonjol (Dark Theme Blue/Cyan) */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284C7 0%, #06B6D4 100%) !important; /* Softer medical blue/cyan gradient */
        color: white !important;
        border: none !important;
        border-radius: 50px !important; /* Bentuk Kapsul/Pill */
        padding: 16px 24px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        box-shadow: 0 10px 20px rgba(6, 182, 212, 0.15) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 25px rgba(6, 182, 212, 0.3) !important;
    }

    /* Kustomisasi Warna Slider & Toggles (make them cyan) */
    .stSlider > div > div > div > div { background-color: #06B6D4 !important; }
    .stToggle input[type="checkbox"]:checked ~ div { background-color: #06B6D4 !important; }
    
    /* Styling Hasil (Alert Cards - Styled for Dark Mode) */
    .alert-danger {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 6px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        color: #F87171;
    }
    .alert-success {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 6px solid #10B981;
        border-radius: 12px;
        padding: 24px;
        color: #6EE7B7;
    }
    
    h1, h2, h3, h4, h5, h6 { color: #F3F4F6 !important; }
    p, label, li, ul, span, stCaption { color: #D1D5DB !important; }
    
    hr { border-color: #374151; }
    
    /* Plotly Chart styling fixes for dark background */
    [data-testid="stPlotlyChart"] {
        border-radius: 16px;
        background-color: #1F2937 !important;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MEMUAT MODEL MACHINE LEARNING
# ==========================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('models/svm_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_models()

# ==========================================
# 4. HEADER APLIKASI
# ==========================================
st.markdown("<h1 class='main-title'>🩺 Sistem Analitis Mata Kering Klinis</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>Platform diagnostik awal berbasis pemodelan data klinis (SVM RBF) untuk menganalisis risiko <i>Dry Eye Disease</i> berdasarkan parameter gaya hidup dan pemeriksaan gejala fisik.</p>", unsafe_allow_html=True)

# ==========================================
# 5. FORM INPUT DATA KLINIS (Card Layout Dark Mode)
# ==========================================
col_kiri, col_kanan = st.columns(2, gap="large")

# Bagian Kiri: Gaya Hidup
with col_kiri:
    with st.container(border=True):
        st.markdown("<h4 style='color: #F3F4F6; margin-bottom: 20px;'>👤 Parameter Gaya Hidup & Vitals</h4>", unsafe_allow_html=True)
        
        age = st.number_input("📅 Usia Pasien (Tahun)", min_value=10, max_value=80, value=24, step=1)
        st.markdown("<br>", unsafe_allow_html=True)
        
        sleep_duration = st.slider("🛌 Rata-rata Durasi Tidur Harian (Jam)", 3.0, 12.0, 7.0, step=0.5)
        screen_time = st.slider("💻 Durasi Paparan Layar Digital (Jam/Hari)", 0.5, 16.0, 7.5, step=0.5)
        
        stress_level = st.select_slider(
            "🤯 Indeks Stres Psikologis",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            value=4,
            format_func=lambda x: "Santai (1)" if x == 1 else ("Sangat Tertekan (10)" if x == 10 else str(x))
        )

# Bagian Kanan: Gejala Klinis
with col_kanan:
    with st.container(border=True):
        st.markdown("<h4 style='color: #F3F4F6; margin-bottom: 20px;'>👁️ Pemeriksaan Gejala Fisik</h4>", unsafe_allow_html=True)
        st.info("💡 **Aktifkan sakelar (toggle)** jika pasien mengeluhkan kondisi ini:")
        
        st.markdown("<br>", unsafe_allow_html=True)
        eye_strain = st.toggle("⚡ Mata sering terasa tegang / lelah (Eye-Strain)?", value=False)
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        
        redness = st.toggle("🔴 Area putih mata sering tampak kemerahan (Redness)?", value=False)
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        
        itchiness = st.toggle("😣 Mata sering terasa gatal, perih, atau seperti berpasir?", value=False)
        st.markdown("<br>", unsafe_allow_html=True)

# Konversi input UI ke format Biner
strain_biner = 1 if eye_strain else 0
redness_biner = 1 if redness else 0
itchiness_biner = 1 if itchiness else 0

# ==========================================
# 6. LOGIKA PREDIKSI & TOMBOL EKSEKUSI
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)

col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
with col_btn2:
    # Menggunakan type="primary" agar CSS terbaru kita teraplikasi dengan sempurna
    start_prediction = st.button("PROSES DATA DIAGNOSTIK SEKARANG", type="primary", use_container_width=True)

# ==========================================
# 7. HASIL ANALISIS (Muncul Setelah Klik)
# ==========================================
if start_prediction:
    if model is None or scaler is None:
        st.error("🚨 Kesalahan Sistem: Berkas model belum terdeteksi. Pastikan file .pkl tersedia di folder models.")
    else:
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Animasi Loading Medis
        with st.spinner('Mengevaluasi jaringan neural dan memetakan vektor data klinis...'):
            st.progress(0)
            time.sleep(1.5)
            st.progress(100)
        
        # Prediksi ML
        input_pasien = np.array([[age, sleep_duration, stress_level, screen_time, strain_biner, redness_biner, itchiness_biner]])
        input_terstandar = scaler.transform(input_pasien)
        
        hasil_prediksi = model.predict(input_terstandar)
        probabilitas = model.predict_proba(input_terstandar)[0]
        skor_risiko = probabilitas[1] * 100 
        
        st.markdown("<h2 style='text-align: center; color: #F3F4F6; margin-bottom: 30px;'>📊 Laporan Hasil Diagnostik</h2>", unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1, 1.2], gap="large")
        
        # --- Kolom Kiri: Gauge Meter Interaktif Styled Dark Mode (Plotly) ---
        with res_col1:
            with st.container(border=True):
                # Warna Gauge disesuaikan tema gelap
                gauge_bar_color = "#EF4444" if hasil_prediksi[0] == 1 else "#06B6D4" # Cyan safe, Red danger
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = skor_risiko,
                    number = {'suffix': "%", 'font': {'size': 40, 'color': '#F3F4F6', 'family': 'Plus Jakarta Sans'}},
                    title = {'text': "Probabilitas Sindrom Mata Kering", 'font': {'size': 14, 'color': '#9CA3AF'}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4B5563"},
                        'bar': {'color': gauge_bar_color, 'thickness': 0.7},
                        'bgcolor': "#374151", # Latar Gauge Gray
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.2)"}, # Soft Green (Safe)
                            {'range': [30, 60], 'color': "rgba(250, 204, 21, 0.2)"}, # Soft Yellow (Warning)
                            {'range': [60, 100], 'color': "rgba(239, 68, 68, 0.2)"} # Soft Red (Danger)
                        ],
                    }
                ))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#F3F4F6"}, margin=dict(l=20, r=20, t=30, b=20), height=260)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # --- Kolom Kanan: Rekomendasi Medis (Dark Mode Alerts) ---
        with res_col2:
            if hasil_prediksi[0] == 1:
                st.markdown(f"""
                <div class="alert-danger">
                    <h3 style="color: #FCA5A5; margin-top: 0;">⚠️ BERISIKO TINGGI (High Risk)</h3>
                    <p style="font-size: 16px; margin-bottom: 15px;">Profil visual Anda memiliki korelasi kuat terhadap Sindrom Mata Kering (Dry Eye Disease).</p>
                    <b style="color: #F87171;">Tindakan Preventif Medis:</b>
                    <ul style="margin-top: 8px;">
                        <li>Segera kurangi intensitas penggunaan gawai. Terapkan aturan jeda mata aturan 20-20-20.</li>
                        <li>Gunakan obat tetes air mata buatan (<i>artificial tears</i>) jika mata terasa sangat kering/perih.</li>
                        <li>Konsultasikan kondisi organ mata Anda lebih lanjut ke Dokter Spesialis Mata (Sp.M).</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-success">
                    <h3 style="color: #6EE7B7; margin-top: 0;">✅ KONDISI SEHAT (Low Risk)</h3>
                    <p style="font-size: 16px; margin-bottom: 15px;">Sistem tidak mendeteksi indikasi kuat terhadap Sindrom Mata Kering. Parameter profil Anda berada dalam batas toleransi.</p>
                    <b style="color: #10B981;">Saran Pemeliharaan:</b>
                    <ul style="margin-top: 8px;">
                        <li>Pertahankan durasi tidur harian Anda ({sleep_duration} jam/hari).</li>
                        <li>Tetap batasi paparan radiasi sinar biru berlebih jika sedang bekerja.</li>
                        <li>Jaga hidrasi tubuh dengan mengonsumsi air putih yang cukup (min. 2L per hari).</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# 8. FOOTER DOKUMEN
# ==========================================
st.markdown("""
<div style='text-align: center; margin-top: 60px; color: #4B5563; font-size: 13px;'>
    Sistem Analitis Medis Digital • Didukung oleh Support Vector Machine (SVM)<br>
    <i>Penafian: Sistem ini hanya memberikan deteksi awal dan bukan merupakan diagnosis medis mutlak.</i>
</div>
""", unsafe_allow_html=True)
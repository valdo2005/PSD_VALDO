import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model(model_type):
    if model_type == "1_hari":
        model = joblib.load('knn_model_day2.pkl')
        scaler = joblib.load('scaler_day2.pkl')
    else:  # 3_hari
        model = joblib.load('knn_model_day3.pkl')
        scaler = joblib.load('scaler_day3.pkl')
    return model, scaler

@st.cache_data
def load_thresholds():
    return joblib.load('thresholds.pkl')

st.set_page_config(page_title="Prediksi NO₂ Pematangsiantar", page_icon="🌫️")
st.title("🌫️ Prediksi Kadar NO₂ - Pematangsiantar")
st.caption("Prediksi kadar NO₂ troposfer (mol/m²) menggunakan model KNN berdasarkan data Sentinel-5P.")

# Pilihan model prediksi
prediksi_type = st.selectbox(
    "Pilih Jenis Prediksi:",
    ("1_hari", "3_hari"),
    format_func=lambda x: "Prediksi 1 Hari Ke Depan" if x == "1_hari" else "Prediksi 3 Hari Ke Depan"
)

model, scaler = load_model(prediksi_type)
thresholds = load_thresholds()

def get_kategori(nilai):
    """Kategorisasi berdasarkan threshold"""
    if nilai <= thresholds['low']:
        return "🟢 **RENDAH**", "Kadar NO₂ rendah, kualitas udara baik."
    elif nilai <= thresholds['medium']:
        return "🟡 **SEDANG**", "Kadar NO₂ sedang, masih dalam batas aman."
    else:
        return "🔴 **TINGGI**", "Kadar NO₂ tinggi, waspadai kualitas udara!"

# Input berbeda berdasarkan jenis prediksi
if prediksi_type == "1_hari":
    # Prediksi 1 hari menggunakan lag 2
    col1, col2 = st.columns(2)
    t2 = col1.number_input("NO₂ (t-2)", value=0.000028, format="%.6f")
    t1 = col2.number_input("NO₂ (t-1)", value=0.000030, format="%.6f")
else:
    # Prediksi 3 hari menggunakan lag 3
    col1, col2, col3 = st.columns(3)
    t3 = col1.number_input("NO₂ (t-3)", value=0.000025, format="%.6f")
    t2 = col2.number_input("NO₂ (t-2)", value=0.000028, format="%.6f")
    t1 = col3.number_input("NO₂ (t-1)", value=0.000030, format="%.6f")

if st.button("Prediksi", type="primary"):
    if prediksi_type == "1_hari":
        X = pd.DataFrame({'t-1': [t1], 't-2': [t2]})
        X_scaled = scaler.transform(X)
    else:
        X = pd.DataFrame({'t-3': [t3], 't-2': [t2], 't-1': [t1]})
        X_scaled = scaler.transform(X)
    
    y_pred = model.predict(X_scaled)[0]
    
    st.success("### Hasil Prediksi")
    
    num_predictions = 1 if prediksi_type == "1_hari" else 3
    cols = st.columns(num_predictions)
    
    for i in range(num_predictions):
        col = cols[i] if num_predictions > 1 else cols[0]
        with col:
            nilai_molm2 = y_pred[i] if num_predictions > 1 else y_pred
            kategori, keterangan = get_kategori(nilai_molm2)
            
            st.metric(f"NO₂ (t+{i+1})", f"{nilai_molm2:.6f} mol/m²")
            st.markdown(kategori)
            st.caption(keterangan)

with st.expander("📋 Informasi Threshold"):
    st.markdown(f"""    
    **Kategori Kadar NO₂:**
    - 🟢 **RENDAH**: ≤ {thresholds['low']:.6f} mol/m²
    - 🟡 **SEDANG**: {thresholds['low']:.6f} - {thresholds['medium']:.6f} mol/m²
    - 🔴 **TINGGI**: > {thresholds['medium']:.6f} mol/m²
    
    **Statistik Data:**
    - Minimum: {thresholds['min']:.6f} mol/m²
    - Maksimum: {thresholds['max']:.6f} mol/m²
    - Rata-rata: {thresholds['mean']:.6f} mol/m²
    """)

st.divider()
st.caption("Model: KNN Regression (3 lag) | Data: 268 observasi | Metode Threshold: Quartile")
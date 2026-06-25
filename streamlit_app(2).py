
import streamlit as st
import pandas as pd
import joblib

# Memuat model, scaler, dan rentang fitur yang disimpan
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_ranges = joblib.load('feature_ranges.pkl') # Memuat rentang fitur

# Judul aplikasi
st.title('Aplikasi Prediksi Kualitas Air')
st.write('Masukkan parameter kualitas air untuk memprediksi apakah air aman untuk dikonsumsi.')

# Mendapatkan nama fitur dalam urutan yang digunakan untuk pelatihan
original_feature_names = scaler.feature_names_in_ # Mengasumsikan scaler memiliki atribut feature_names_in_

input_data = {}
for feature in original_feature_names:
    min_val = feature_ranges[feature]['min']
    max_val = feature_ranges[feature]['max']
    mean_val = feature_ranges[feature]['mean']

    # Menggunakan st.number_input untuk setiap input, menyediakan nilai min, max, dan nilai rata-rata default
    input_data[feature] = st.sidebar.number_input(
        f'Masukkan {feature}',
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(mean_val) # Default ke nilai rata-rata
    )

# Mengubah data input menjadi DataFrame, memastikan urutan kolom sesuai dengan data pelatihan
input_df = pd.DataFrame([input_data], columns=original_feature_names)

# Menskalakan data input
scaled_input = scaler.transform(input_df)

# Membuat prediksi
if st.button('Prediksi Keamanan Air'):
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.subheader('Hasil Prediksi:')
    if prediction[0] == 1:
        st.success('Air diprediksi **AMAN** untuk dikonsumsi.')
    else:
        st.error('Air diprediksi **TIDAK AMAN** untuk dikonsumsi.')

    st.write(f"Kepercayaan Diri (Aman): {prediction_proba[0][1]:.2f}")
    st.write(f"Kepercayaan Diri (Tidak Aman): {prediction_proba[0][0]:.2f}")

st.sidebar.markdown('---')
st.sidebar.markdown('Dikembangkan oleh Nama/Organisasi Anda')

import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="churn_recommendation.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load scaler
scaler = joblib.load("scaler.pkl")

# Judul Aplikasi
st.title("Prediksi Churn Nasabah Kartu Kredit")
st.write("Masukkan data nasabah untuk memprediksi apakah nasabah akan churn (berhenti menggunakan layanan).")

# Form input user
row_number  = st.number_input("row_number", min_value=0, max_value=900, value=650)
customer_id = st.number_input("customer_id", min_value=0, max_value=900, value=650)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age = st.number_input("Usia", min_value=18, max_value=100, value=40)
tenure = st.number_input("Lama Menjadi Nasabah (Tahun)", min_value=0, max_value=10, value=3)
balance = st.number_input("Saldo", min_value=0.0, max_value=300000.0, value=50000.0)
num_products = st.number_input("Jumlah Produk", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Memiliki Kartu Kredit", options=[0, 1])
is_active_member = st.selectbox("Status Keaktifan Nasabah", options=[0, 1])
estimated_salary = st.number_input("Estimasi Gaji", min_value=0.0, max_value=300000.0, value=100000.0)

# Input sebagai array
input_data = np.array([[row_number,customer_id,credit_score, age, tenure, balance, num_products, has_cr_card, is_active_member, estimated_salary]])

# Prediksi saat tombol diklik
if st.button("Prediksi Churn"):
    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = np.argmax(prediction)
    probability = prediction[0][predicted_class]

    if predicted_class == 1:
        st.error(f"⚠️ Nasabah kemungkinan besar akan **CHURN**. (Probabilitas: {probability:.2f})")
    else:
        st.success(f"✅ Nasabah diprediksi **TIDAK churn**. (Probabilitas: {probability:.2f})")

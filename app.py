import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Memuat model terbaik
model = joblib.load('model.pkl')

# Memuat data untuk pengkodean dan penskalaan
data = pd.read_csv('onlinefoods.csv')

# Daftar kolom yang diperlukan selama pelatihan
required_columns = ['Age', 'Gender', 'Marital Status', 'Occupation', 'Monthly Income', 'Educational Qualifications', 'Family size', 'latitude', 'longitude', 'Pin code', 'Feedback']

# Pastikan hanya kolom yang diperlukan ada
data = data[required_columns]

# Pra-pemrosesan data
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = data[column].astype(str)
    le.fit(data[column])
    data[column] = le.transform(data[column])
    label_encoders[column] = le

scaler = StandardScaler()
numeric_features = ['Age', 'Family size', 'latitude', 'longitude', 'Pin code']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Fungsi untuk memproses input pengguna
def preprocess_input(user_input):
    processed_input = {col: [user_input.get(col, 'Unknown')] for col in required_columns}
    for column in label_encoders:
        if column in processed_input:
            input_value = processed_input[column][0]
            if input_value in label_encoders[column].classes_:
                processed_input[column] = label_encoders[column].transform([input_value])
            else:
                # Jika nilai tidak dikenal, berikan nilai default seperti -1
                processed_input[column] = [-1]
    processed_input = pd.DataFrame(processed_input)
    processed_input[numeric_features] = scaler.transform(processed_input[numeric_features])
    return processed_input

# CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F0F8FF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #4B0082;
        text-align: center;
        margin-bottom: 20px;
    }
    h3 {
        color: #4B0082;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #4B0082;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #6A0DAD;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 20px;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #D3D3D3;
        width: 100%;
    }
    .stTextInput {
        margin-bottom: 20px;
        padding: 10px;
        border-radius: 6px;
        border: 1px solid #D3D3D3;
        width: 100%;
    }
    .centered-text {
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-container {
        background-color: #FFF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Antarmuka Streamlit
st.title("Prediksi Output Customer Online Food")

st.markdown("""
    <div class="centered-text">
        <h3>Web ini akan menentukan apakah seorang pelanggan akan membeli makanan secara online atau tidak</h3>
    </div>
""", unsafe_allow_html=True)

# Input pengguna
with st.form(key='input_form'):
    age = st.number_input('Age', min_value=18, max_value=100, help="Masukkan usia Anda")
    gender = st.selectbox('Gender', ['Male', 'Female'], help="Pilih jenis kelamin Anda")
    marital_status = st.selectbox('Marital Status', ['Single', 'Married'], help="Pilih status perkawinan Anda")
    occupation = st.selectbox('Occupation', ['Student', 'Employee', 'Self Employed'], help="Pilih pekerjaan Anda")
    monthly_income = st.selectbox('Monthly Income', ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'], help="Pilih pendapatan bulanan Anda")
    educational_qualifications = st.selectbox('Educational Qualifications', ['Under Graduate', 'Graduate', 'Post Graduate'], help="Pilih tingkat pendidikan Anda")
    family_size = st.number_input('Family size', min_value=1, max_value=20, help="Masukkan jumlah anggota keluarga Anda")
    latitude = st.number_input('Latitude', format="%f", help="Masukkan koordinat latitude")
    longitude = st.number_input('Longitude', format="%f", help="Masukkan koordinat longitude")
    pin_code = st.number_input('Pin code', min_value=100000, max_value=999999, help="Masukkan kode pos Anda")
    feedback = st.selectbox('Feedback', ['Negative', 'Positive'], help="Pilih umpan balik Anda")

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    user_input = {
        'Age': age,
        'Gender': gender,
        'Marital Status': marital_status,
        'Occupation': occupation,
        'Monthly Income': monthly_income,
        'Educational Qualifications': educational_qualifications,
        'Family size': family_size,
        'latitude': latitude,
        'longitude': longitude,
        'Pin code': pin_code,
        'Feedback': feedback
    }
    user_input_processed = preprocess_input(user_input)
    try:
        prediction = model.predict(user_input_processed)
        # Mengubah hasil prediksi menjadi 'Yes' atau 'No'
        prediction_label = 'Yes' if prediction[0] == 1 else 'No'
        with st.container():
            st.write(f'*Prediction:* {prediction_label}', unsafe_allow_html=True)
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

# Tambahkan elemen HTML untuk output
st.markdown("""
    <div class="centered-text">
        <h3>Output Prediksi</h3>
        <p>Hasil prediksi akan ditampilkan di sini.</p>
    </div>
""", unsafe_allow_html=True)

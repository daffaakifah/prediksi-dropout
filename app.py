import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Daftar lengkap fitur
ALL_FEATURES = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Previous_qualification_grade', 'Nationality', 'Mothers_occupation',
    'Fathers_occupation', 'Admission_grade', 'Displaced',
    'Educational_special_needs', 'Debtor', 'Tuition_fees_up_to_date',
    'Gender', 'Scholarship_holder', 'Age_at_enrollment', 'International',
    'Unemployment_rate', 'Inflation_rate', 'GDP',
    'Attendance_Time', 'Previous_education', 'Mothers_education',
    'Fathers_education', '1st_year_approved_unit', '1st_year_enrolled_unit',
    '1st_year_evaluated_unit', '1st_year_credited_unit',
    '1st_year_unevaluated_unit', '1st_year_unit_grade',
    '1st_year_completion_unit_rate', '1st_year_success_evaluation_rate'
]

NUMERIC_FEATURES = [
     'Application_mode', 'Application_order', 'Previous_qualification_grade', 
     'Admission_grade', 'Age_at_enrollment', 'Unemployment_rate', 
     'Inflation_rate', 'GDP', '1st_year_approved_unit', 
     '1st_year_enrolled_unit', '1st_year_evaluated_unit', '1st_year_credited_unit', 
     '1st_year_unevaluated_unit', '1st_year_unit_grade', '1st_year_completion_unit_rate', 
     '1st_year_success_evaluation_rate'
]

CATEGORICAL_FEATURES = [f for f in ALL_FEATURES if f not in NUMERIC_FEATURES]

# Load model
model = joblib.load('model.pkl')

@st.cache_data
def load_data(path='data_enrolled.csv'):
    df = pd.read_csv(path)
    return df

def add_missing_features_with_defaults(df):
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            if feature in CATEGORICAL_FEATURES:
                df[feature] = 'unknown'
            else:
                df[feature] = 0
    df = df[ALL_FEATURES]
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def main():
    st.title("📊 Prediksi Dropout Mahasiswa - Jaya Jaya Institute")

    df_enrolled = load_data()
    df_enrolled = add_missing_features_with_defaults(df_enrolled)

    menu = st.sidebar.selectbox("Pilih Menu", ["Prediksi Enrolled", "Prediksi Input Pengguna"])

    if menu == "Prediksi Enrolled":
        st.write("Jaya Jaya Institut merupakan institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan dikenal luas berkat reputasinya dalam mencetak lulusan berkualitas. Namun, di balik prestasi tersebut, institusi ini menghadapi tantangan serius berupa tingginya angka mahasiswa yang tidak menyelesaikan studi alias dropout. Fenomena ini menjadi perhatian utama karena dapat berdampak negatif terhadap citra institusi, efektivitas proses belajar-mengajar, serta efisiensi penggunaan sumber daya. Tingginya tingkat dropout tidak hanya merugikan institusi, tetapi juga para mahasiswa yang gagal meraih gelar dan masa depan yang lebih baik. Oleh karena itu, Jaya Jaya Institut berupaya untuk mendeteksi secara dini mahasiswa yang berpotensi mengalami dropout. Dengan deteksi dini, institusi dapat memberikan intervensi dan bimbingan khusus agar mahasiswa tersebut tetap termotivasi dan mampu menyelesaikan pendidikannya.")
        st.write("Prediksi Dropout Mahasiswa berdasarkan data yang Sedang Enrolled")

        pred_proba = model.predict_proba(df_enrolled)[:, 1]
        # Gunakan threshold 0.5 untuk klasifikasi
        preds = np.where(pred_proba < 0.5, 1, 0)

        df_enrolled['Prediksi_Dropout'] = np.where(preds == 1, 'Dropout', 'Tidak Dropout')
        df_enrolled['Probabilitas_Graduate'] = pred_proba.round(3)

        st.dataframe(df_enrolled[[
            '1st_year_completion_unit_rate', '1st_year_enrolled_unit', 'Tuition_fees_up_to_date', '1st_year_approved_unit', 'Admission_grade', 'Scholarship_holder',  
            '1st_year_unit_grade', '1st_year_success_evaluation_rate', 'Prediksi_Dropout', 'Probabilitas_Graduate'
        ]])

        st.markdown("""
        **Rekomendasi Aksi untuk Mahasiswa yang Diprediksi Dropout:**
        - **Intervensi Akademik:** Berikan bimbingan belajar tambahan, konsultasi akademik, dan monitoring progres.
        - **Dukungan Finansial:** Tinjau kemungkinan beasiswa atau bantuan biaya kuliah.
        - **Konseling Psikologis:** Sediakan layanan konseling untuk mengatasi masalah pribadi atau motivasi.
        - **Peningkatan Keterlibatan:** Ajak mahasiswa untuk bergabung dalam kegiatan kampus dan kelompok belajar.
        """)

    elif menu == "Prediksi Input Pengguna":
        st.write("Prediksi Dropout Mahasiswa Berdasarkan Input Pengguna")

        with st.form("input_form"):
            completion_rate = st.number_input("1st Year Completion Unit Rate (0-1)", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
            enrolled_unit = st.number_input("1st Year Enrolled Unit", min_value=0, max_value=100, value=30)
            approved_unit = st.number_input("1st Year Approved Unit", min_value=0, max_value=100, value=28)
            admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
            unit_grade = st.number_input("1st Year Unit Grade", min_value=0.0, max_value=100.0, value=14.0, step=0.1)
            success_eval_rate = st.number_input("1st Year Success Evaluation Rate (0-1)", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
            tuition_fees = st.selectbox("Tuition Fees Up To Date", options=['Yes', 'No'])
            scholarship_holder = st.selectbox("Scholarship Holder", options=['Yes', 'No'])

            submitted = st.form_submit_button("Prediksi")

        if submitted:
            input_data = pd.DataFrame({
                '1st_year_completion_unit_rate': [completion_rate],
                '1st_year_enrolled_unit': [enrolled_unit],
                '1st_year_approved_unit': [approved_unit],
                'Admission_grade': [admission_grade],
                '1st_year_unit_grade': [unit_grade],
                '1st_year_success_evaluation_rate': [success_eval_rate],
                'Tuition_fees_up_to_date': [tuition_fees],
                'Scholarship_holder': [scholarship_holder]
            })

            input_data = add_missing_features_with_defaults(input_data)

            prediction_proba = model.predict_proba(input_data)[0]
            graduate_prob = prediction_proba[1]
            status = "Dropout" if graduate_prob < 0.5 else "Tidak Dropout"

            st.subheader("Hasil Prediksi")
            st.write(f"Prediksi mahasiswa **{status}** dengan probabilitas:")
            st.write(f"- Tidak Dropout: {graduate_prob:.2f}")
            st.write(f"- Dropout: {prediction_proba[0]:.2f}")

            if graduate_prob < 0.5:
                st.markdown("""
                ### ⚠️ Rekomendasi Aksi untuk Mahasiswa yang Berpotensi Dropout:
                - **Intervensi Akademik:** Berikan bimbingan belajar tambahan, konsultasi akademik, dan monitoring progres.
                - **Dukungan Finansial:** Tinjau kemungkinan beasiswa atau bantuan biaya kuliah.
                - **Konseling Psikologis:** Sediakan layanan konseling untuk mengatasi masalah pribadi atau motivasi.
                - **Peningkatan Keterlibatan:** Ajak mahasiswa untuk bergabung dalam kegiatan kampus dan kelompok belajar.
                """)

st.sidebar.write("""
    - **Dataset:** [Jaya Jaya Institut's Students' Performance](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/data.csv)
    - **Email:** daffaakifahbalqis01@gmail.com
    - **Dicoding Username:** daffabalqis
""")

if __name__ == "__main__":
    main()

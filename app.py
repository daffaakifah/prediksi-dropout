import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# --- Daftar lengkap fitur ---
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

# --- Tentukan fitur numerik dan kategorikal ---
NUMERIC_FEATURES = [
     'Application_mode', 'Application_order', 'Previous_qualification_grade', 
     'Admission_grade', 'Age_at_enrollment', 'Unemployment_rate', 
     'Inflation_rate', 'GDP', '1st_year_approved_unit', 
     '1st_year_enrolled_unit', '1st_year_evaluated_unit', '1st_year_credited_unit', 
     '1st_year_unevaluated_unit', '1st_year_unit_grade', '1st_year_completion_unit_rate', 
     '1st_year_success_evaluation_rate'
]

CATEGORICAL_FEATURES = [f for f in ALL_FEATURES if f not in NUMERIC_FEATURES]

# --- Fungsi load model dan preprocessor ---
@st.cache_data
def load_model_and_preprocessor(model_path='model.pkl', preprocessor_path='preprocessor.pkl'):
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Debug tipe preprocessor
    print(f"Loaded preprocessor type: {type(preprocessor)}")
    from sklearn.base import TransformerMixin
    if not isinstance(preprocessor, TransformerMixin):
        raise TypeError("Preprocessor yang dimuat bukan transformer yang valid.")
    
    return model, preprocessor

# --- Fungsi load data enrolled ---
@st.cache_data
def load_data(path='data_enrolled.csv'):
    df = pd.read_csv(path)
    return df

# --- Fungsi untuk menambahkan fitur default ---
def add_missing_features_with_defaults(df):
    # Tambahkan kolom yang hilang dengan nilai default
    for feature in ALL_FEATURES:
        if feature not in df.columns:
            if feature in CATEGORICAL_FEATURES:
                df[feature] = 'unknown'
            else:
                df[feature] = 0
    
    # Pastikan urutan kolom sama dengan ALL_FEATURES
    df = df[ALL_FEATURES]
    
    # Konversi tipe data untuk memastikan kompatibilitas
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    return df

# --- Fungsi utama aplikasi Streamlit ---
def main():
    st.title("📊 Prediksi Dropout Mahasiswa - Jaya Jaya Institute")

    # Load model dan preprocessor
    model, preprocessor = load_model_and_preprocessor()
    df_enrolled = load_data()

    # Tambahkan fitur default pada data enrolled
    df_enrolled = add_missing_features_with_defaults(df_enrolled)

    # Sidebar menu
    menu = st.sidebar.selectbox("Pilih Menu", ["Prediksi Enrolled", "Prediksi Manual"])

    if menu == "Prediksi Enrolled":
        st.header("Prediksi Dropout Mahasiswa yang Sedang Enrolled")

        # Preprocess data menggunakan preprocessor yang sama dengan training
        X_enrolled = preprocessor.transform(df_enrolled)

        # Prediksi
        preds = model.predict(X_enrolled)
        pred_proba = model.predict_proba(X_enrolled)[:, 1]

        # Tambahkan hasil prediksi ke dataframe
        df_enrolled['Prediksi_Dropout'] = np.where(preds == 1, 'Dropout', 'Tidak Dropout')
        df_enrolled['Probabilitas_Dropout'] = pred_proba.round(3)

        # Tampilkan tabel hasil prediksi dengan fitur penting saja
        st.dataframe(df_enrolled[[
            '1st_year_completion_unit_rate', '1st_year_enrolled_unit', 'Admission_grade',
            'Tuition_fees_up_to_date', 'Scholarship_holder', 'Prediksi_Dropout', 'Probabilitas_Dropout'
        ]])

        st.markdown("""
        **Rekomendasi Aksi untuk Mahasiswa yang Diprediksi Dropout:**
        - **Intervensi Akademik:** Berikan bimbingan belajar tambahan, konsultasi akademik, dan monitoring progres.
        - **Dukungan Finansial:** Tinjau kemungkinan beasiswa atau bantuan biaya kuliah.
        - **Konseling Psikologis:** Sediakan layanan konseling untuk mengatasi masalah pribadi atau motivasi.
        - **Peningkatan Keterlibatan:** Ajak mahasiswa untuk bergabung dalam kegiatan kampus dan kelompok belajar.
        """)

    elif menu == "Prediksi Manual":
        st.header("Prediksi Dropout Mahasiswa Berdasarkan Input Manual")

        # Form input user untuk 8 fitur penting
        with st.form("input_form"):
            completion_rate = st.number_input("1st Year Completion Unit Rate (0-1)", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
            enrolled_unit = st.number_input("1st Year Enrolled Unit", min_value=0, max_value=100, value=30)
            approved_unit = st.number_input("1st Year Approved Unit", min_value=0, max_value=100, value=28)
            admission_grade = st.number_input("Admission Grade", min_value=0.0, max_value=20.0, value=15.0, step=0.1)
            unit_grade = st.number_input("1st Year Unit Grade", min_value=0.0, max_value=20.0, value=14.0, step=0.1)
            success_eval_rate = st.number_input("1st Year Success Evaluation Rate (0-1)", min_value=0.0, max_value=1.0, value=0.75, step=0.01)
            tuition_fees = st.selectbox("Tuition Fees Up To Date", options=['Yes', 'No'])
            scholarship_holder = st.selectbox("Scholarship Holder", options=['Yes', 'No'])

            submitted = st.form_submit_button("Prediksi")

        if submitted:
            # Buat dataframe input user dengan 8 fitur
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

            # Tambahkan fitur default untuk fitur yang tidak diinput user
            input_data = add_missing_features_with_defaults(input_data)

            # Preprocess input menggunakan preprocessor yang sama
            X_input = preprocessor.transform(input_data)

            # Prediksi
            prediction = model.predict(X_input)
            prediction_proba = model.predict_proba(X_input)

            # Tampilkan hasil
            status = "Dropout" if prediction[0] == 1 else "Tidak Dropout"
            st.subheader("Hasil Prediksi")
            st.write(f"Prediksi mahasiswa **{status}** dengan probabilitas:")
            st.write(f"- Dropout: {prediction_proba[0][1]:.2f}")
            st.write(f"- Tidak Dropout: {prediction_proba[0][0]:.2f}")

    # Tampilkan feature importance (opsional)
    if st.checkbox("Tampilkan Feature Importance"):
        import matplotlib.pyplot as plt
        import seaborn as sns

        try:
            # Asumsi model adalah pipeline dengan classifier
            classifier = model.named_steps['classifier']
            importances = classifier.feature_importances_
            
            # Dapatkan nama fitur dari preprocessor
            num_features = preprocessor.transformers_[0][2]
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES)
            feature_names = list(num_features) + list(cat_features)

            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            fi_df = fi_df.sort_values(by='Importance', ascending=False)

            st.write(fi_df.head(20))

            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(x='Importance', y='Feature', data=fi_df.head(20), ax=ax)
            ax.set_title("Top 20 Feature Importance")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal menampilkan feature importance: {e}")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page config
st.set_page_config(page_title="PCOS Prediction ANN", page_icon="🩺", layout="wide")

# Judul aplikasi
st.title('🩺 PCOS Prediction using Artificial Neural Network')

# Menu navigasi
menu = ["📚 Tentang PCOS", "📊 Data Exploration", "🤖 Model Training", "🔮 Prediction"]
choice = st.sidebar.selectbox("Menu", menu)

# image = Image.open('images/cacar air.JPG')
# st.image(image,caption='cacar air pada manusia', use_column_width=True)

# Fungsi untuk membangun model ANN
def build_model(input_shape, dropout_rate=0.2, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model

# Fungsi untuk memproses data
def process_data(df):
    features = [' Age (yrs)', 'Cycle(R/I)', 'Cycle length(days)',
                'Pregnant(Y/N)', 'LH(mIU/mL)', 'Hip(inch)', 'Waist(inch)',
                'Waist:Hip Ratio', 'AMH(ng/mL)', 'Weight gain(Y/N)', 
                'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Hair loss(Y/N)', 
                'Pimples(Y/N)', 'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 
                'Follicle No. (L)', 'Follicle No. (R)']
    
    target = 'PCOS (Y/N)'
    
    X = df[features]
    y = df[target]
    X = X.fillna(X.mean())
    
    return X, y

# Halaman Tentang PCOS
if choice == "📚 Tentang PCOS":
    st.header("Yang perlu kamu ketahui")
    st.write("""
    **Organisasi Kesehatan Dunia (WHO) mengatakan bahwa sekitar 117 juta perempuan pernah menderita PCOS,
             yang merupakan kurang lebih 3,5% dari semua perempuan di seluruh dunia.
             Sebagian besar penderita PCOS tidak mengetahui bahwa dirinya terkena penyakit tersebut karena minimnya pengetahuan.
             Penyakit ini dapat menjadi sangat berbahaya jika tidak segera mendapatkan perawatan medis,
             sehingga perlu adanya deteksi awal agar penyakit PCOS dapat ditangani segera sebelum sampai pada tahap yang lebih serius.
             Salah satu studi di RS Cipto Mangunkusumo menemukan 105 perempuan menderita PCOS.
             Dengan 94,2% pasien mengalami gejala amenore (tidak mengalami menstruasi),
             dan 32,4% mengalami hirsutisme atau pertumbuhan rambut secara berlebihan. 45,7% pasien dalam rentang usia 26-30 tahun.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gejala PCOS")
        st.write("""
        Gejala penyakit PCOS antara lain ketidakseimbangan hormon, menstruasi yang tidak teratur (amenore),
                 hiperandrogenisme (hormon androgen yang meningkat), hirsutisme (pertumbuhan rambut yang berlebihan di area tubuh),
                 alopecia (kerontokan rambut), acne (jerawat), dan skin darkening[19]. PCOS juga dapat menyebabkan diabetes tipe 2,
                 obesitas, dan penyakit kardiovaskular (penyakit terkait dengan pembuluh darah dan jantung).
                 Menurut dosen Spesialis Obstetri dan Ginekologi Fakultas Kedokteran Universitas Airlangga (UNAIR) Dr. Sri Ratna Dwiningsih dr SpOG (K),
                 faktor-faktor yang dapat menyebabkan PCOS termasuk obesitas, kurang aktivitas fisik, riwayat keluarga PCOS,
                 atau paparan intrauterin di dalam rahim. Selain itu, bahan kimia lingkungan seperti bisphenol A, dioxins,
                 dan triclosan dianggap dapat mengganggu sistem endokrin dan menyebabkan PCOS[20].
        """)
    
    with col2:
        st.subheader("Diagnosis PCOS")
        st.write("""
        Dengan perkembangan ilmu teknologi saat ini, teknologi komputer dan kesehatan dapat menjadi solusi dalam mendiagnosa sebuah penyakit,
        terutama menggunakan teknik data mining dan machine learning. Penelitian mengenai deteksi awal penyakit menggunakan metode data mining banyak dilakukan.
        Salah satu metode data mining seperti prediksi atau dikenal sebagai forecasting, digunakan untuk memprediksi nilai atau kemungkinan yang terjadi di masa depan.
        Beberapa algoritma data mining yang digunakan dalam prediksi, yaitu Support Vector Machine (SVM), Neural Network dan Linear Regression.
        Seperti penelitian yang dilakukan oleh Subrato Bharati dkk tahun 2020 dalam memprediksi penyakit PCOS menggunakan algoritma Gradient Boosting, Random Forest, Logistic Regression, dan Hybrid Random Forest and Logistic Regression (RFLR).
        Di antara algoritma tersebut, Hybrid Random Forest and Logistic Regression (RFLR) menghasilkan akurasi 91,01%.
        Artificial Neural Network (ANN) termasuk dalam algoritma machine learning untuk menangani masalah prediksi,
        algoritma ini bekerja dengan pola komputasi seperti berdasarkan sistem jaringan saraf biologis serta tersusun dari beberapa proses yang terhubung atau disebut dengan neuron.
        Penelitian yang dilakukan oleh Muhammad Resha dkk tahun 2023 membuktikan bahwa algoritma ANN unggul dalam memprediksi penyakit tuberkulosis dengan tingkat akurasi 97,59%.
        Penelitian terkait algoritma Artificial Neural Network juga dilakukan oleh Serin Wulandari dkk dalam menguji dataset penyakit stroke dan mendapatkan hasil akurasi 94,83%[9]. 

        """)
        
        st.image("https://images.ctfassets.net/6m9bd13t776q/1HrQ0j3zQ0K3ZQ7QJQ7QJQ/6e9e9e9e9e9e9e9e9e9e9e9e9e9e9e9/pcos.jpg", 
                 caption="Ilustrasi Ovarium Polikistik", width=300)
    
    st.markdown("---")
    st.subheader("Sumber Informasi Terpercaya")
    st.write("""
    - [Kementerian Kesehatan RI](https://www.kemkes.go.id/)
    - [American College of Obstetricians and Gynecologists](https://www.acog.org/)
    - [Mayo Clinic](https://www.mayoclinic.org/)
    - [PCOS Awareness Association](https://www.pcosaa.org/)
    """)

# Halaman Data Exploration
elif choice == "📊 Data Exploration":
    st.header("Data Exploration")
    
    st.sidebar.header("⚙️ Pengaturan Data")
    uploaded_file = st.sidebar.file_uploader("Upload dataset Anda (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data berhasil dimuat!")
        
        X, y = process_data(df)
        
        st.write("### Dataset Awal (5 baris pertama)")
        st.write(df.head())
        
        st.write("### Distribusi Kelas")
        fig1, ax1 = plt.subplots()
        y.value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('Distribusi Kelas PCOS (Y/N)')
        st.pyplot(fig1)
        
        st.session_state.X = X
        st.session_state.y = y
    else:
        st.warning("Silakan upload file CSV untuk memulai")

# Halaman Model Training
elif choice == "🤖 Model Training":
    st.header("Model Training")
    
    if 'X' not in st.session_state or 'y' not in st.session_state:
        st.warning("Silakan upload dan proses data terlebih dahulu di halaman Data Exploration")
        st.stop()
    
    X = st.session_state.X
    y = st.session_state.y
    
    st.sidebar.header("⚙️ Parameter Model")
    epochs = st.sidebar.slider("Jumlah Epoch", 10, 500, 100)
    batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)
    test_size = st.sidebar.slider("Persentase Data Testing", 0.1, 0.5, 0.4)
    validation_split = st.sidebar.slider("Validation Split", 0.1, 0.3, 0.1)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2, step=0.05)
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
    
    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Normalisasi
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.session_state.scaler = scaler
    
    # Bangun model
    model = build_model(
        X_train_scaled.shape[1], 
        dropout_rate=dropout_rate,
        learning_rate=learning_rate
    )
    
    # Latih model
    if st.button("🚀 Train Model"):
        with st.spinner("Melatih model..."):
            history = model.fit(
                X_train_scaled, y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            st.session_state.model = model
            
            st.success("Pelatihan selesai!")
            
            # Plot training history
            st.write("### Training History")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.plot(history.history['accuracy'])
            ax1.plot(history.history['val_accuracy'])
            ax1.set_title('Model Accuracy')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.legend(['Train', 'Validation'], loc='upper left')
            
            ax2.plot(history.history['loss'])
            ax2.plot(history.history['val_loss'])
            ax2.set_title('Model Loss')
            ax2.set_ylabel('Loss')
            ax2.set_xlabel('Epoch')
            ax2.legend(['Train', 'Validation'], loc='upper left')
            
            st.pyplot(fig)
            
            # Evaluasi
            st.write("### Evaluasi pada Test Set")
            loss, accuracy, precision, recall = model.evaluate(X_test_scaled, y_test, verbose=0)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Loss", f"{loss:.4f}")
            col2.metric("Accuracy", f"{accuracy:.4f}")
            col3.metric("Precision", f"{precision:.4f}")
            col4.metric("Recall", f"{recall:.4f}")
            
            # Confusion matrix
            st.write("### Confusion Matrix")
            y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Classification report
            st.write("### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.table(pd.DataFrame(report).transpose())

# Halaman Prediction
elif choice == "🔮 Prediction":
    st.header("Prediction")
    
    if 'model' not in st.session_state or 'scaler' not in st.session_state:
        st.warning("Silakan latih model terlebih dahulu di halaman Model Training")
        st.stop()
    
    model = st.session_state.model
    scaler = st.session_state.scaler
    
    st.write("### Prediksi dari Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (yrs)", min_value=10, max_value=60, value=25)
        cycle = st.selectbox("Cycle(R/I)", options=["R", "I"])
        cycle_length = st.number_input("Cycle length(days)", min_value=1, max_value=90, value=28)
        pregnant = st.selectbox("Pregnant(Y/N)", options=["Y", "N"])
        lh = st.number_input("LH(mIU/mL)", min_value=0.0, max_value=50.0, value=5.0)
        hip = st.number_input("Hip(inch)", min_value=20.0, max_value=60.0, value=35.0)
    
    with col2:
        waist = st.number_input("Waist(inch)", min_value=20.0, max_value=60.0, value=30.0)
        waist_hip_ratio = st.number_input("Waist:Hip Ratio", min_value=0.5, max_value=1.5, value=0.85)
        amh = st.number_input("AMH(ng/mL)", min_value=0.0, max_value=20.0, value=3.0)
        weight_gain = st.selectbox("Weight gain(Y/N)", options=["Y", "N"])
        hair_growth = st.selectbox("hair growth(Y/N)", options=["Y", "N"])
        skin_darkening = st.selectbox("Skin darkening (Y/N)", options=["Y", "N"])
    
    with col3:
        hair_loss = st.selectbox("Hair loss(Y/N)", options=["Y", "N"])
        pimples = st.selectbox("Pimples(Y/N)", options=["Y", "N"])
        fast_food = st.selectbox("Fast food (Y/N)", options=["Y", "N"])
        exercise = st.selectbox("Reg.Exercise(Y/N)", options=["Y", "N"])
        follicle_l = st.number_input("Follicle No. (L)", min_value=0, max_value=50, value=5)
        follicle_r = st.number_input("Follicle No. (R)", min_value=0, max_value=50, value=5)
    
    # Konversi input
    cycle = 1 if cycle == "R" else 0
    pregnant = 1 if pregnant == "Y" else 0
    weight_gain = 1 if weight_gain == "Y" else 0
    hair_growth = 1 if hair_growth == "Y" else 0
    skin_darkening = 1 if skin_darkening == "Y" else 0
    hair_loss = 1 if hair_loss == "Y" else 0
    pimples = 1 if pimples == "Y" else 0
    fast_food = 1 if fast_food == "Y" else 0
    exercise = 1 if exercise == "Y" else 0
    
    input_data = np.array([[age, cycle, cycle_length, pregnant, lh, hip, waist,
                          waist_hip_ratio, amh, weight_gain, hair_growth, skin_darkening,
                          hair_loss, pimples, fast_food, exercise, follicle_l, follicle_r]])
    
    if st.button("🔮 Predict"):
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        probability = prediction[0][0]
        result = "PCOS (Positive)" if probability > 0.5 else "Normal (Negative)"
        
        st.success(f"### Hasil Prediksi: {result}")
        st.info(f"Probabilitas: {probability:.4f}")
        
        fig, ax = plt.subplots()
        ax.bar(['Normal', 'PCOS'], [1-probability, probability], color=['green', 'red'])
        ax.set_ylim(0, 1)
        ax.set_title('Prediction Probability')
        st.pyplot(fig)
    
    st.write("---")
    st.write("### Prediksi dari File")
    
    pred_file = st.file_uploader("Upload file untuk prediksi (CSV)", type=["csv"], key="pred_file")
    
    if pred_file is not None:
        pred_df = pd.read_csv(pred_file)
        
        try:
            X_pred, _ = process_data(pred_df)
            X_pred_scaled = scaler.transform(X_pred)
            predictions = model.predict(X_pred_scaled)
            pred_classes = (predictions > 0.5).astype(int)
            
            pred_df['PCOS_Prediction'] = pred_classes
            pred_df['PCOS_Probability'] = predictions
            
            st.success("Prediksi selesai!")
            st.write("### Hasil Prediksi")
            st.write(pred_df)
            
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Prediksi",
                data=csv,
                file_name='pcos_predictions.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error dalam memproses file: {str(e)}")

# Tampilkan model summary jika model sudah dilatih
if 'model' in st.session_state and choice != "📚 Tentang PCOS":
    with st.expander("ℹ️ Model Summary"):
        st.text("Arsitektur Model ANN:")
        st.session_state.model.summary(print_fn=lambda x: st.text(x))
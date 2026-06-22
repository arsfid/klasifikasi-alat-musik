import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Klasifikasi Instrumen Musik",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Klasifikasi Instrumen Musik")
st.write("Upload gambar instrumen musik untuk diprediksi oleh model CNN.")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("instrument_cnn_model.h5")

# =========================
# LOAD CLASS NAMES
# =========================
@st.cache_resource
def load_classes():
    with open("class_names.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
class_names = load_classes()

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(image):

    # SESUAIKAN DENGAN MODELMU
    img = image.convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img)

    img_array = img_array.astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# =========================
# UPLOAD FILE
# =========================
uploaded_file = st.file_uploader(
    "Pilih gambar...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    if st.button("Prediksi"):

        with st.spinner("Memproses gambar..."):

            img = preprocess_image(image)

            prediction = model.predict(img)

            pred_index = np.argmax(prediction)

            confidence = float(np.max(prediction))

            predicted_class = class_names[pred_index]

            st.success(
                f"Hasil Prediksi: **{predicted_class}**"
            )

            st.write(
                f"Confidence: **{confidence*100:.2f}%**"
            )

            st.progress(confidence)

            st.subheader("Probabilitas Semua Kelas")

            for i, cls in enumerate(class_names):
                st.write(
                    f"{cls}: {prediction[0][i]*100:.2f}%"
                )
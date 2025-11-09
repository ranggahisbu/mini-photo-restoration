import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Photo Restoration App", page_icon="🪄", layout="centered")

st.title("🪄 Aplikasi Restorasi Foto Lama")
st.write("Perbaiki foto lama dari noise, goresan, blur, dan pudar dengan teknik preprocessing citra dasar.")

uploaded_file = st.file_uploader("Unggah foto lama kamu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = np.array(Image.open(uploaded_file))
    st.image(img, caption="Foto Asli", use_container_width=True)

    # --- 1️⃣ Representasi Citra (Grayscale)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # --- 2️⃣ Filtering (Median + Gaussian Blur)
    # Gunakan kombinasi untuk hilangkan bintik halus tapi tetap tajam
    median_filtered = cv2.medianBlur(gray, 3)
    gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)

    # --- 3️⃣ Histogram Equalization (meningkatkan kontras)
    equalized = cv2.equalizeHist(gaussian_filtered)

    # --- 4️⃣ Operasi Geometri (Crop / Rotate)
    st.write("### ✂️ Pengaturan Geometri")
    rotate_angle = st.slider("Putar Gambar (°)", -45, 45, 0)
    crop_x = st.slider("Crop dari kiri (%)", 0, 50, 0)
    crop_y = st.slider("Crop dari atas (%)", 0, 50, 0)

    # Rotasi gambar
    (h, w) = equalized.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(equalized, M, (w, h))

    # Crop gambar
    x_start = int(w * (crop_x / 100))
    y_start = int(h * (crop_y / 100))
    cropped = rotated[y_start:, x_start:]

    # --- Konversi ke BGR sebelum detailEnhance (karena image sekarang grayscale)
    color_restored = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

    # --- Post-processing ringan untuk hasil halus (tanpa overprocess)
    restored = cv2.detailEnhance(color_restored, sigma_s=10, sigma_r=0.15)

    # Gabung tampilan
    st.write("### 🔧 Hasil Restorasi")
    st.image(restored, caption="Foto Setelah Restorasi", use_container_width=True)

    # Tombol simpan
    result_img = Image.fromarray(restored)
    st.download_button(
        label="💾 Unduh Hasil Restorasi",
        data=cv2.imencode('.jpg', restored)[1].tobytes(),
        file_name="restored_photo.jpg",
        mime="image/jpeg"
    )
else:
    st.info("Silakan unggah foto lama untuk memulai restorasi.")

st.markdown("---")
st.caption("🧠 Menggunakan teknik preprocessing: Grayscale, Filtering, Histogram Equalization, dan Geometri")

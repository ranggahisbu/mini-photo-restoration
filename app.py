import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Photo Restoration App", page_icon="🪄", layout="centered")

st.title("🪄 Aplikasi Restorasi Foto Lama")
st.write("Perbaiki foto lama dari noise, goresan, blur, dan pudar dengan hasil yang natural (tidak overprocess).")

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
    median_filtered = cv2.medianBlur(gray, 3)
    gaussian_filtered = cv2.GaussianBlur(median_filtered, (5, 5), 0)

    # --- 3️⃣ Histogram Equalization pakai CLAHE (biar gak over)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gaussian_filtered)

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

    # --- Denoising ringan biar hasil lembut
    denoised = cv2.fastNlMeansDenoising(cropped, h=10)

    # --- Konversi ke BGR sebelum detailEnhance
    color_restored = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    # --- DetailEnhance dengan pengaturan lembut
    restored_soft = cv2.detailEnhance(color_restored, sigma_s=5, sigma_r=0.1)

    # --- Blend hasil supaya natural (70% hasil + 30% original)
    blended = cv2.addWeighted(color_restored, 0.3, restored_soft, 0.7, 0)

    # Gabung tampilan
    st.write("### 🔧 Hasil Restorasi (Natural Mode)")
    st.image(blended, caption="Foto Setelah Restorasi", use_container_width=True)

    # Tombol simpan
    result_img = Image.fromarray(blended)
    st.download_button(
        label="💾 Unduh Hasil Restorasi",
        data=cv2.imencode('.jpg', blended)[1].tobytes(),
        file_name="restored_photo_natural.jpg",
        mime="image/jpeg"
    )
else:
    st.info("Silakan unggah foto lama untuk memulai restorasi.")

st.markdown("---")
st.caption("🧠 Teknik: Grayscale, Filtering, CLAHE, Denoising, dan Operasi Geometri (Natural Restoration Mode)")

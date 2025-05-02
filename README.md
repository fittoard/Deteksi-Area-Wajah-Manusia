# Deteksi Area Wajah Manusia Pada Citra Berwarna Berbasis Segmentasi Warna YCbCr dan Operasi Morfologi Citra

berikut merupakan code untuk mencoba nya

```
# Upload gambar
from google.colab import files
uploaded = files.upload()

# Import library
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Ambil nama file
filename = list(uploaded.keys())[0]

# Baca dan proses gambar
img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Ekstraksi komponen Cb dan Cr
Y, Cr, Cb = cv2.split(img_ycbcr)

# Deteksi kulit berdasarkan threshold YCbCr
skin_mask = cv2.inRange(img_ycbcr, (0, 135, 85), (255, 180, 135))

# Thresholding dari grayscale
_, thresh_mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Operasi morfologi (erosi, opening, dilasi)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(skin_mask, kernel, iterations=1)
opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
dilated = cv2.dilate(opened, kernel, iterations=1)

# Masking hasil morfologi ke gambar asli
masked = cv2.bitwise_and(img_rgb, img_rgb, mask=dilated)

# Tampilkan hasil seperti layout paper (2 kolom x 3 baris)
titles = [
    "Citra YCbCr", "Citra hasil grayscale",
    "Area kulit terdeteksi", "Citra hasil thresholding",
    "Citra hasil morfologi", "Hasil masking image"
]

images = [
    cv2.cvtColor(img_ycbcr, cv2.COLOR_YCrCb2RGB),  # Konversi balik YCbCr ke RGB
    img_gray,
    skin_mask,
    thresh_mask,
    dilated,
    masked
]

# Plot layout 2x3
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(3, 2, i + 1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

Proyek ini merupakan implementasi sederhana dari sistem deteksi wajah pada gambar berwarna dengan pendekatan berbasis warna kulit. Metode yang digunakan mengandalkan pemisahan warna menggunakan ruang warna YCbCr, lalu dilanjutkan dengan serangkaian operasi morfologi untuk memperjelas area wajah. Semua proses ini dilakukan di lingkungan Google Colab menggunakan Python, dengan bantuan pustaka OpenCV, NumPy, dan Matplotlib.

Setelah pengguna mengunggah gambar, citra akan diproses dan dikonversi ke tiga format: RGB untuk tampilan visual, grayscale untuk thresholding berbasis intensitas, serta YCbCr yang berguna untuk mendeteksi area kulit. Deteksi warna kulit dilakukan dengan threshold pada nilai Cb dan Cr, yang umum mewakili rona warna kulit manusia. Hasil segmentasi ini kemudian diproses menggunakan morfologi seperti erosi, opening, dan dilasi untuk membersihkan noise serta memperkuat area wajah yang terdeteksi.

Langkah terakhir adalah menampilkan hasil akhir dalam bentuk masking pada gambar asli, hanya menyisakan bagian wajah. Seluruh tahapan proses ini divisualisasikan dalam satu grid 3x2, mulai dari hasil konversi warna, deteksi kulit, hingga citra akhir. Proyek ini cocok sebagai bahan eksperimen atau pembelajaran awal dalam bidang pengolahan citra digital, khususnya bagi yang ingin memahami cara kerja segmentasi warna tanpa menggunakan model machine learning yang kompleks.


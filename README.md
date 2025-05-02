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

📌 Deteksi Area Wajah Menggunakan Segmentasi Warna YCbCr dan Operasi Morfologi
Proyek ini merupakan implementasi sistem deteksi wajah pada citra berwarna menggunakan pendekatan segmentasi warna kulit pada ruang warna YCbCr yang dipadukan dengan operasi morfologi citra, sesuai dengan metode dari paper "Deteksi Area Wajah Manusia Berbasis Segmentasi Warna YCbCr dan Operasi Morfologi".

🔧 Teknologi dan Library
Python 3 (Google Colab)

OpenCV

NumPy

Matplotlib

PIL (Python Imaging Library)

⚙️ Alur Proses
Upload Gambar
Pengguna mengunggah gambar wajah berwarna dari lokal menggunakan google.colab.files.upload().

Konversi Warna

Gambar dikonversi dari BGR ke RGB untuk ditampilkan.

Juga dikonversi ke grayscale untuk proses thresholding berbasis intensitas.

Kemudian dikonversi ke YCbCr untuk segmentasi warna kulit.

Segmentasi Warna Kulit
Komponen Cb dan Cr dari ruang warna YCbCr digunakan untuk membedakan area kulit dari bukan kulit melalui threshold (0, 135, 85) hingga (255, 180, 135).

Thresholding Grayscale
Metode Otsu digunakan untuk menghasilkan threshold otomatis pada citra grayscale (sebagai pembanding proses binerisasi).

Operasi Morfologi

Erosi: Mengurangi noise dengan menghapus piksel pinggir.

Opening: Membersihkan objek kecil yang tidak diinginkan.

Dilasi: Memperluas area hasil deteksi untuk memperjelas fitur wajah.

Masking
Hasil akhir morfologi digunakan untuk menampilkan hanya area wajah dari gambar asli dengan cv2.bitwise_and().

Visualisasi
Semua tahap ditampilkan dalam layout 3 baris × 2 kolom:

Citra YCbCr

Citra Grayscale

Deteksi Area Kulit

Hasil Thresholding

Hasil Operasi Morfologi

Masking Akhir (deteksi wajah)

📁 Struktur Output
Setiap hasil ditampilkan menggunakan matplotlib dalam format visual yang memudahkan evaluasi proses per tahap. Kode ini sangat cocok untuk digunakan dalam:

Tugas akhir/paper pengolahan citra

Pembelajaran segmentasi warna kulit

Dasar deteksi wajah tanpa deep learning

💡 Catatan
Akurasi deteksi bergantung pada pencahayaan dan latar belakang.

Nilai threshold YCbCr dapat disesuaikan untuk warna kulit yang berbeda.

Direkomendasikan menggunakan gambar wajah dengan latar sederhana untuk hasil optimal.




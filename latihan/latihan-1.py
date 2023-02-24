import cv2

# buat variable yang berisi gambar didalamnya
image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_COLOR)
"""
cv2.IMREAD_COLOR/1 -> Membaca Warna tanpa Transparan
cv2.IMREAD_GRAYSCALE/0 -> Membaca dalam Grayscale
cv2.IMREAD_UNCHANGED/-1-> Membaca dalam kondisi Asli
"""

# tampilkan pada widows
cv2.imshow('gambar1', image)

# seperti readkey
cv2.waitKey(0)

# menutup semua widows yang ada
cv2.destroyAllWindows()
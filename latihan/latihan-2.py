# # Load gambar
# ==================================================================== grey scale
# Image = cv2.imread('../imgs/dumy-img-1.jpg')

# # cari tau ukuran H wan W
# H,W = Image.shape[:2]
# gray = np.zeros((H,W), np.uint8)

# # lakukan perulangan kepada setiap pixel pada gambar
# for i in range(H):
#     for j in range(W):
#         # mengubah citra ke greyscale
#         # f(x,y) = 0.299R + 0.587G + 0.114B
#         gray[i,j] = np.clip(0.299 * Image[i, j, 0]+
#                             0.587 * Image[i, j, 1]+
#                             0.114 * Image[1, j, 2], 0, 255)
# # tampilkan gambar
# cv2.imshow("ORI", Img)

# # alternative ubah ke grayscale
# Image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)
# cv2.imshow("ORI", Img)

# # ubah ke grayscale
# Image = cv2.imread('../imgs/dumy-img-1.jpg', 0)

# ================================================ Brightness
# # melakukan brighness kepada setiap pixel pada array
# H,W = Image.shape[:2]
# # tentukan value birghtness nya
# BRIGHTNES = 80
# for i in range(H):
#     for j in range(W):
#         # nilai pixel greyscale di cari
#         a = Image.item(i, j)
#         # nilai pixel di tambahkan dengan nilai brightnes
#         # f(x, y)’ = f(x, y) + b 
#         b = np.clip(a + BRIGHTNES, 0, 255)
#         Image.itemset((i, j), b)

# ================================================== 

# Image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)

# H,W = Image.shape[:2]
# BRIGHTNES = 80
# for i in range(H):
#     for j in range(W):
#         a = Image.item(i, j)
#         b = np.clip(a + BRIGHTNES, 0, 255)
#         Image.itemset((i, j), b)

# cv2.imshow("ORI", Image)

# ===================================================== Contras

# Image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)

# H,W = Image.shape[:2]

# CONTRAS = 1.5
# for i in range(H):
#     for j in range(W):
#         a = Image.item(i, j)
#         b = np.clip(a * CONTRAS, 0, 255)
#         Image.itemset((i, j), b)

# ===================================================== 

Image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)

# H,W = Image.shape[:2]
# CONTRAS = 1
# for i in range(H):
#     for j in range(W):
#         a = Image.item(i, j)
#         b = np.clip(a * CONTRAS, 0, 255)
#         Image.itemset((i, j), b)

# cv2.imshow("ORI", Image)

# ======================================================== Contras Streching

# H,W = Image.shape[:2]
# minV = np.min(self.Image)
# maxV = np.max(self.Image)

# for i in range(H):
#     for j in range(W):
#         a = Image.item(i, j)
#         # merata-ratakan nilai kontras
#         b = float(a - minV) / (maxV - minV) * 255
#         Image.itemset((i, j), b)

# cv2.imshow("ORI", Image)

# ======================================================== negative Image

# Image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)

# H,W = Image.shape[:2]
# MAXIMUM_INTENSITY = 255

# for i in range(H):
#     for j in range(W):
#         a = Image.item(i, j)
#         # menentukan nilai masing2 pixel negative
#         # f(x, y)’ = 255 – f(x, y) 
#         b = math.ceil(MAXIMUM_INTENSITY - a)
#         Image.itemset((i, j), b)

# cv2.imshow("ORI", Image)

# ========================================================== Biner

# Image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)

# H,W = Image.shape[:2]
# THRESHOLD = 180

# for i in range(H):
#     for j in range(W):
#         a = Image.item(i, j)
#         # bila nilainya adalah 100, maka pixel nya akan bernilai 0
#         if(a == THRESHOLD):
#             b = 0
#         # bila kurang dari threshold, maka akan bernilai 1
#         elif(a < THRESHOLD):
#             b = 1
#         # selebihnya akan bernilai 255
#         else:
#             b = 255

#         Image.itemset((i, j), b)

# cv2.imshow("Hasil", Image)

# ============================================================ Grey Histogram

# plt.hist(Image.ravel(), 255, [0, 255])
# plt.show()

# =========================================================== RGB Histogram

# Image = cv2.imread('../imgs/dumy-img-1.jpg')

# # nilai yang akan di tampilkan pada historgam
# color = ('b', 'g', 'r') 

# # i adalah index dari color; col adalah value 'b', 'g', 'r' nya
# for i,col in enumerate(color):
#     # system perhitungan cv2
#     histo = cv2.calcHist([self.Image], [i], None, [255], [0,255])
#     # ploting pada histogram
#     plt.plot(histo, color = col) 
#     # mengatur batas sumbu x
#     plt.xlim([0,256])

# ============================================================= Equalization

# # membuat self.Image menjadi BGR lagi
# Image = cv2.imread('../imgs/dumy-img-1.jpg')
# cv2.imshow("Ori", Image)

# hist, bins = np.histogram(Image.flatten(), 256, [0, 256]) #mengubah array img menjadi 1 dimensi
# cdf = hist.cumsum() # menentukan jumlah kumulatif array pada bagian tertentu
# cdf_normalized = cdf * hist.max() / cdf.max() # untuk normalisasi
# cdf_m = np.ma.masked_equal(cdf, 0) # memasking nilai array dengan yang di berikan
# cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) # melakukan perhitungan
# cdf = np.ma.filled(cdf_m, 0).astype('uint8') # mengisi array dengan nilai skalar
# Image = cdf[Image] # mengganti nilai array image menjadi nilai komulatif

# cv2.imshow("Hasil", Image)

# # ploting histogram
# plt.plot(cdf_normalized, color='b') # melakukan ploting sesuai normalisasi
# plt.hist(Image.flatten(), 256, [0, 256], color='r') # membuat histogram sesuai dengan nilai array gambar
# plt.xlim([0, 256]) # mengatur batas sumbu x
# plt.legend(('cdf', 'histogram'), loc='upper left') # membuat text di histogramnya
# plt.show()

# =================================================================== Translasi

# Image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)

# h, w = Image.shape[:2] # membagi dan mendapatkan height dan width
# quarter_h,quarter_w = h/4, w/4 # menentukan bakal kaya gimana ntr translasiinya
# T = np.float32([[1,0,quarter_w],[0,1,quarter_h]])
# img = cv2.warpAffine(Image, T, (w,h))

# cv2.imshow("Hasil", img)

# =================================================================== Rotasi
# Image = cv2.imread('../imgs/dumy-img-1.jpg')

# degree = 180
# h, w = Image.shape[:2] # mendapatkan bentuk width dan height nya gambar
# rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7) # jari-jari di kali derajat

# # mencari matriks rotasi titik 0,0 dan 0,1
# cos = np.abs(rotationMatrix[0, 0])
# sin = np.abs(rotationMatrix[0, 1])

# nW = int((h * sin) + (w * cos))
# nH = int((h * cos) + (w * sin))

# # mencari matriks rotasi titik 0,2 dan 1,2
# rotationMatrix[0, 2] += (nW / 2) - w / 2
# rotationMatrix[1, 2] += (nH / 2) - h / 2

# # lakukan rotasi pada self.Image
# rot_image = cv2.warpAffine(Image, rotationMatrix, (h, w))

# cv2.imshow("Hasil", rot_image)

# ======================================================================= Zoom
# Image = cv2.imread('../imgs/dumy-img-1.jpg')
# cv2.imshow('Ori', resize_img)

# skala = 2

# resize_img = cv2.resize(Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
# cv2.imshow('Zoom ', resize_img)

# ===================================================================== Crop

# Image = cv2.imread('../imgs/dumy-img-1.jpg')
# cv2.imshow('Ori', resize_img)

# crop_atas_vertical = (20 / 100) * H # nilai 20 nggk boleh lebih dari 100 - crop_bawah_vertical
# crop_atas_horizontal = (10 / 100) * W # nilai 10 nggk boleh lebih dari 100 - crop_bawah_horizontal
# crop_bawah_vertical = H - (30 / 100) * H # nilai 30 nggk boleh lebih dari 100 - crop_atas_vertical
# crop_bawah_horizontal = W - (20 / 100) * W # nilai 20 nggk boleh lebih dari 100 - crop_atas_horizontal

# cropped_image = self.Image[int(crop_atas_horizontal):int(crop_bawah_vertical), int(crop_atas_vertical):int(crop_bawah_horizontal)]
# cv2.imshow("cropped", cropped_image)

# ===================================================================== Aritmatika

# Image1 = cv2.cvtColor('../imgs/dumy-img-1.jpg', cv2.COLOR_BGR2GRAY)
# H, W = Image1.shape[:2]
# img = cv2.cvtColor('../imgs/dumy-img-2.jpg', cv2.COLOR_BGR2GRAY)
# Image2 = cv2.resize(img, (W, H))

# hasil = Image1 + Image2
# cv2.imshow("hasil pertambahan", hasil) 

# hasil = Image1 - Image2
# cv2.imshow("hasil pengurangan", hasil) 

# hasil = Image1 * Image2
# cv2.imshow("hasil perkalian", hasil) 

# hasil = Image1 / Image2
# cv2.imshow("hasil perkalian", hasil) 

# ========================================================================= Boolean

# Image1 = cv2.cvtColor('../imgs/dumy-img-1.jpg', cv2.COLOR_BGR2GRAY)
# H, W = Image1.shape[:2]
# img = cv2.cvtColor('../imgs/dumy-img-2.jpg', cv2.COLOR_BGR2GRAY)
# Image2 = cv2.resize(img, (W, H))

# hasil = cv2.bitwise_and(Image1, Image2)
# cv2.imshow("hasil AND", hasil) 


# hasil = cv2.bitwise_or(Image1, Image2)
# cv2.imshow("hasil OR", hasil) 

# hasil = cv2.bitwise_xor(Image1, Image2)
# cv2.imshow("hasil XOR", hasil) 

# ========================================================================== Konvolusi

# baca ukuran dimensi citra
H_citra, W_citra = arrycitra.shape

# baca ukuran dimensi karnel
H_karnel, W_karnel = arrykarnel.shape

# meenutukan titik tengah
H = H_karnel // 2
W = W_karnel // 2   

out = np.zeros((H_citra, W_citra))

# menggeser karnel konvolusi
for i in range(H + 1, H_citra - H):
    for j in range(W + 1, W_citra - W):
        sum = 0
        for k in range(-H, H):
            for l in range(-W, W):
                citra_value = arrycitra[i + k, j + l]
                kernel_value = arrykarnel[H + k, W + l]
                sum += citra_value * kernel_value
            out[i, j] = copy.copy(sum)






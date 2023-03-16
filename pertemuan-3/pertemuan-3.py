#Import Library
"""
import sys
import cv2
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog, Qapplication, QMainWindow
from PyQt5.uic import loadUi

"""
import sys
import cv2
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
import numpy as np
import math 
import tkinter as tk
from tkinter import filedialog
import copy

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("GUI.ui", self)
        self.Image = None

        # Button 
        self.button_loadCitra.clicked.connect(self.fungsi)
        self.button_prosesCitra.clicked.connect(self.greyscale)

        # File Action
        self.actionSave.triggered.connect(self.save)
        self.actionLoad.triggered.connect(self.openClick)

        # operasi Titik
        self.actionOperasi_pencerahan.triggered.connect(self.brightness)
        self.actionSimple_contras.triggered.connect(self.contras)
        self.actionContras_streching.triggered.connect(self.contrasstreching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)
        self.actionHistogram_greyscale.triggered.connect(self.greyHistogram)
        self.actionHistogram_RGB.triggered.connect(self.rgbHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.equalzationHistogram)

        # Operasi Geometri
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.action_90_Derajat.triggered.connect(self.rotasiMin90derajat) 
        self.action45_Derajat.triggered.connect(self.rotasi45derajat)
        self.action_45_Derajat.triggered.connect(self.rotasiMin45derajat)
        self.action180_Derajat.triggered.connect(self.rotasi180derajat)
        self.action200.triggered.connect(self.zoom200)
        self.action300.triggered.connect(self.zoom300)
        self.action400.triggered.connect(self.zoom400)
        self.action75.triggered.connect(self.zoom75)
        self.action50.triggered.connect(self.zoom50)
        self.action25.triggered.connect(self.zoom25)
        self.actionCrop.triggered.connect(self.cropImage)

        # Aritmatika
        self.actionPertambahan_2.triggered.connect(self.pertambahan)
        self.actionPengurangan.triggered.connect(self.pengurangan)
        self.actionPerkalian.triggered.connect(self.perkalian)
        self.actionPembagian.triggered.connect(self.pembagian)
        
        # Boolean
        self.actionAND.triggered.connect(self.opand)
        self.actionOR.triggered.connect(self.opor)
        self.actionXOR.triggered.connect(self.opxor)

        # Operasi Spasial
        self.actionKarnel1.triggered.connect(self.konvolusi1)   
        self.actionKarnel2.triggered.connect(self.konvolusi2)   
        self.actionMeanFilter_2.triggered.connect(self.meanfilter)   
        self.actionGaussianFilter.triggered.connect(self.gaussianfilter)   
        self.actionKarnel_i.triggered.connect(self.sharpening_i)   
        self.actionKarnel_ii.triggered.connect(self.sharpening_ii)   
        self.actionKarnel_iii .triggered.connect(self.sharpening_iii)   
        self.actionKarnel_iv.triggered.connect(self.sharpening_iv)   
        self.actionKarnel_v.triggered.connect(self.sharpening_v)   
        self.actionKarnel_vi.triggered.connect(self.sharpening_vi)   
        self.actionLaplace.triggered.connect(self.laplacefilter)   
        self.actionMedian_filter.triggered.connect(self.medianfilter)   
        self.actionMax_Filter.triggered.connect(self.maximumfilter)   
        self.actionMin_Filter.triggered.connect(self.minimumfilter)   


    # fungsi menampilkan citra normal
    def fungsi(self):
        self.Image = cv2.imread('../imgs/dumy-img-1.jpg')
        self.displayImage()
    
    # fungsi memproses citra greyscale
    def greyscale(self):
        try :
            H,W = self.Image.shape[:2]
            gray = np.zeros((H,W), np.uint8)
            for i in range(H):
                for j in range(W):
                    # mengubah citra ke greyscale
                    # f(x,y) = 0.299R + 0.587G + 0.114B
                    gray[i,j] = np.clip(0.299 * self.Image[i, j, 0]+
                                        0.587 * self.Image[i, j, 1]+
                                        0.114 * self.Image[1, j, 2], 0, 255)
            self.Image = gray
        except :
            self.Image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)
        
        self.displayImage(2)

    # fungsi greyscale tapi di cerahin
    def brightness(self):
        #Error hendling
        try : 
            # bila display kosong, maka akan di buat gambar greyscale secara manual
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H,W = self.Image.shape[:2]
        BRIGHTNES = self.brightnesSlider.value()
        for i in range(H):
            for j in range(W):
                # nilai pixel greyscale di cari
                a = self.Image.item(i, j)
                # nilai pixel di tambahkan dengan nilai brightnes
                # f(x, y)’ = f(x, y) + b 
                b = np.clip(a + BRIGHTNES, 0, 255)
                self.Image.itemset((i, j), b)
        
        self.displayImage(2)
    
    # fungsi mengatur kontras
    def contras(self):
        #Error hendling
        try : 
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H,W = self.Image.shape[:2]
        CONTRAS = 1 + (self.contrasSlider.value() / 100)
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                # mengatur kontras dengan parameter contras
                # f(x, y)’ = f(x, y) * c
                b = np.clip(a * CONTRAS, 0, 255)
                self.Image.itemset((i, j), b)
        
        self.displayImage(2)
    
    # fungsi mengatur kontras streching
    def contrasstreching(self):
        #Error hendling
        try : 
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H,W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                # merata-ratakan nilai kontras
                b = float(a - minV) / (maxV - minV) * 255
                self.Image.itemset((i, j), b)
        
        self.displayImage(2)

    # fungsi mengatur gambar negative
    def negative(self):
        try : 
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H,W = self.Image.shape[:2]
        MAXIMUM_INTENSITY = 255

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                # menentukan nilai masing2 pixel negative
                # f(x, y)’ = 255 – f(x, y) 
                b = math.ceil(MAXIMUM_INTENSITY - a)
                self.Image.itemset((i, j), b)
        
        self.displayImage(2)
    
    # fungsi mengatur gambar biner
    def biner(self):
        try : 
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H,W = self.Image.shape[:2]
        THRESHOLD = 180

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                # bila nilainya adalah 100, maka pixel nya akan bernilai 0
                if(a == THRESHOLD):
                    b = 0
                # bila kurang dari threshold, maka akan bernilai 1
                elif(a < THRESHOLD):
                    b = 1
                # selebihnya akan bernilai 255
                else:
                    b = 255

                self.Image.itemset((i, j), b)
        
        self.displayImage(2)

    # menampilkan tampilan histogram greyscale
    def greyHistogram(self):
        try :
            H,W = self.Image.shape[:2]
            gray = np.zeros((H,W), np.uint8)
            for i in range(H):
                for j in range(W):
                    # mengubah citra ke greyscale
                    # f(x,y) = 0.299R + 0.587G + 0.114B
                    gray[i,j] = np.clip(0.299 * self.Image[i, j, 0]+
                                        0.587 * self.Image[i, j, 1]+
                                        0.114 * self.Image[1, j, 2], 0, 255)
            self.Image = gray
        except :
            self.Image = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)
        
        self.displayImage(2)

        # untuk menampilkan histogram, menggunakan heist
        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()

    # menampilkan histogram RGB
    def rgbHistogram(self):
        # membuat self.Image menjadi BGR lagi
        self.Image = cv2.imread('../imgs/dumy-img-1.jpg')

        # nilai yang akan di tampilkan pada historgam
        color = ('b', 'g', 'r') 

        # i adalah index dari color; col adalah value 'b', 'g', 'r' nya
        for i,col in enumerate(color):
            # system perhitungan cv2
            histo = cv2.calcHist([self.Image], [i], None, [255], [0,255])
            # ploting pada histogram
            plt.plot(histo, color = col) 
            # mengatur batas sumbu x
            plt.xlim([0,256])
        
        # membuat visualisasi dari histogram
        plt.show()
        self.displayImage(2)

    # meratakan histogram
    def equalzationHistogram(self):
        # membuat self.Image menjadi BGR lagi
        self.Image = cv2.imread('../imgs/dumy-img-1.jpg')

        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256]) #mengubah array img menjadi 1 dimensi
        cdf = hist.cumsum() # menentukan jumlah kumulatif array pada bagian tertentu
        cdf_normalized = cdf * hist.max() / cdf.max() # untuk normalisasi
        cdf_m = np.ma.masked_equal(cdf, 0) # memasking nilai array dengan yang di berikan
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) # melakukan perhitungan
        cdf = np.ma.filled(cdf_m, 0).astype('uint8') # mengisi array dengan nilai skalar
        self.Image = cdf[self.Image] # mengganti nilai array image menjadi nilai komulatif

        self.displayImage(2)
        
        # ploting histogram
        plt.plot(cdf_normalized, color='b') # melakukan ploting sesuai normalisasi
        plt.hist(self.Image.flatten(), 256, [0, 256], color='r') # membuat histogram sesuai dengan nilai array gambar
        plt.xlim([0, 256]) # mengatur batas sumbu x
        plt.legend(('cdf', 'histogram'), loc='upper left') # membuat text di histogramnya
        plt.show()

    # mentranslasikan gambar
    def translasi(self):
        h, w = self.Image.shape[:2] # membagi dan mendapatkan height dan width
        quarter_h,quarter_w = h/4, w/4 # menentukan bakal kaya gimana ntr translasiinya
        T = np.float32([[1,0,quarter_w],[0,1,quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w,h))

        self.Image = img
        self.displayImage(2)

    # Rotasi
    def rotasi90derajat(self):
        self.rotasi(90)
    
    def rotasiMin90derajat(self):
        self.rotasi(-90)

    def rotasi45derajat(self):
        self.rotasi(45)

    def rotasiMin45derajat(self):
        self.rotasi(-45)
    
    def rotasi180derajat(self):
        self.rotasi(180)

    def rotasi(self, degree):
        h, w = self.Image.shape[:2] # mendapatkan bentuk width dan height nya gambar
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7) # jari-jari di kali derajat

        # mencari matriks rotasi titik 0,0 dan 0,1
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # mencari matriks rotasi titik 0,2 dan 1,2
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2

        # lakukan rotasi pada self.Image
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))

        self.Image = rot_image
        self.displayImage(2)

    # Zoom in & Out
    def zoom200(self):
        # memutuskan mau zoom nilainya berapa
        self.zoominout(2, "200%")

    def zoom300(self):
        self.zoominout(3, "300%")

    def zoom400(self):
        self.zoominout(4, "400%")
    
    def zoom75(self):
        self.zoominout(.75, "75%")

    def zoom50(self):
        self.zoominout(.5, "50%")

    def zoom25(self):
        self.zoominout(.25, "25%")

    def zoominout(self, skala, keterangan):
        # melakukan 
        resize_img = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Zoom ' + keterangan, resize_img)
        cv2.waitKey()

    # Crop Image
    def cropImage(self):
        H, W = self.Image.shape[:2]

        try:
            crop_atas_vertical = (self.crop_atas_horizontal.value() / 100) * H
            crop_atas_horizontal = ((100 - self.crop_atas_vertical.value()) / 100) * W
            crop_bawah_vertical = H - (self.crop_bawah_vertical.value() / 100) * H
            crop_bawah_horizontal = W - (self.crop_bawah_horizontal.value() / 100) * W
        
            cropped_image = self.Image[int(crop_atas_horizontal):int(crop_bawah_vertical), int(crop_atas_vertical):int(crop_bawah_horizontal)]
            cv2.imshow("cropped", cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except :
            pass
    
    # Aritmatika
    def aritmatika(self):
        self.Image1 = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        H, W = self.Image1.shape[:2]
        img = cv2.cvtColor(self.open(), cv2.COLOR_BGR2GRAY)
        self.Image2 = cv2.resize(img, (W, H))
        
    def pertambahan(self):
        self.aritmatika()

        hasil = self.Image1 + self.Image2
        cv2.imshow("hasil pertambahan", hasil) 

    def pengurangan(self):
        self.aritmatika()

        hasil = self.Image1 - self.Image2
        cv2.imshow("hasil pengurangan", hasil) 

    def perkalian(self):
        self.aritmatika()

        hasil = self.Image1 * self.Image2
        cv2.imshow("hasil pengurangan", hasil) 

    def pembagian(self):
        self.aritmatika()

        hasil = self.Image1 / self.Image2
        cv2.imshow("hasil pengurangan", hasil) 
    
    # Boolean
    def boolean(self):
        self.Image1 = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        H, W = self.Image1.shape[:2]
        img = cv2.cvtColor(self.open(), cv2.COLOR_BGR2RGB)
        self.Image2 = cv2.resize(img, (W, H))

    def opand(self):
        self.boolean()

        hasil = cv2.bitwise_and(self.Image1, self.Image2)
        cv2.imshow("hasil AND", hasil) 

    def opor(self):
        self.boolean()

        hasil = cv2.bitwise_or(self.Image1, self.Image2)
        cv2.imshow("hasil OR", hasil) 

    def opxor(self):
        self.boolean()

        hasil = cv2.bitwise_xor(self.Image1, self.Image2)
        cv2.imshow("hasil XOR", hasil) 

    # File Composer
    def save(self):
        flname, filter = QFileDialog.getSaveFileName(self, "SaveFile", "C:\\", "Image Files (*.jpg)")
        if flname:
            cv2.imwrite(flname, self.Image)
        else :
            print("error")

    def open(self):
        filename = filedialog.askopenfilename()
        img = cv2.imread(filename)
        return img

    def openClick(self):
        self.Image = self.open()
        self.displayImage()

    # convolusi
    # 1. baca ukuran tinggi dan lebar citra
    # 2. baca ukuran tinggi dan lebar karnel
    # (cara menentukan titik tengahnya adalah membagi tinggi dan lebar lalu di bagi dua dan di bulatkan ke bawah)
    # H = ukuran tinggi karnel / 2
    # W = ukurang lebar karnel / 2
    def math_konvolusi(self, arrycitra, arrykarnel):
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
        
        return out
    
    def konvolusi1(self):
        # mengubah self.image menjadi greyscale
        img = cv2.imread("../imgs/dumy-img-1.jpg")

        # ubah ke gray
        IMGGREY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # analisisi citra sebelum di convolusi
        plt.imshow(IMGGREY, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        KERNEL = np.array([[1,1,1], 
                          [1,1,1],
                          [1,1,1]])
        
        # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
        hasil = self.math_konvolusi(IMGGREY, KERNEL)

        # analisis citra sesudah di konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi')

        plt.show()
        cv2.waitKey()

    def konvolusi2(self):
        # mengubah self.image menjadi greyscale
        img = cv2.imread("../imgs/dumy-img-1.jpg")

        # ubah ke gray
        IMGGREY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # analisisi citra sebelum di convolusi
        plt.imshow(IMGGREY, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        KERNEL = np.array([[6,0,-6], 
                          [6,1,-6],
                          [6,0,-6]])
        
        # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
        hasil = self.math_konvolusi(IMGGREY, KERNEL)

        # analisis citra sesudah di konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi')

        plt.show()
        cv2.waitKey()
    
    def meanfilter(self):
        # Load citra
        img = cv2.imread("../imgs/dumy-img-1.jpg")

        # ubah ke gray
        IMGGREY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # analisisi citra sebelum di convolusi
        plt.imshow(IMGGREY, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        # membuat kernel 3x3
        KERNEL3X3 = np.array([[1/9, 1/9, 1/9], 
                          [1/9, 1/9, 1/9],
                          [1/9, 1/9, 1/9]])
        
        # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
        hasil3x3 = self.math_konvolusi(IMGGREY, KERNEL3X3)

        # analisis citra sesudah di konvolusi
        plt.figure()
        plt.imshow(hasil3x3, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi 3x3')

        # membuat kernel 3x3
        KERNEL2X2 = np.array([[1 / 4, 1 / 4],
                               [1 / 4, 1 / 4]])
        
        # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
        hasil2x2 = cv2.filter2D(IMGGREY, -1, KERNEL2X2)

        # analisis citra sesudah di konvolusi
        plt.figure()
        plt.imshow(hasil2x2, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi 2x2')

        plt.show()
        cv2.waitKey()

    def gaussianfilter(self):
        # Load gambar
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

        # ubah ke gray
        IMGGREY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # analisisi citra sebelum di convolusi
        plt.imshow(IMGGREY, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        # buat kernel
        KERNEL = (1.0 / 345) * np.array([[1, 5, 7, 5, 1],
                                         [5, 20, 33, 20, 5],
                                         [7, 33, 55, 33, 7],
                                         [5, 20, 33, 20, 5],
                                         [1, 5, 7, 5, 1]])
        
        # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
        hasil = self.math_konvolusi(IMGGREY, KERNEL)

        # analisis citra sesudah di konvolusi
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi gaussian')

        plt.show()
        cv2.waitKey()
    
    def sharpening_i(self):
        KERNEL = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                           [-1, -1, -1]])
        self.show_sharpening(KERNEL)

    def sharpening_ii(self):
        KERNEL = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        self.show_sharpening(KERNEL)

    def sharpening_iii(self):
        KERNEL = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        self.show_sharpening(KERNEL)

    def sharpening_iv(self):
        KERNEL = np.array([[1, -2, 1],
                           [-2, 5, -2],
                           [1, -2, 1]])
        self.show_sharpening(KERNEL)

    def sharpening_v(self):
        KERNEL = np.array([[1, -2, 1],
                           [-2, 4, -2],
                           [1, -2, 1]])
        self.show_sharpening(KERNEL)

    def sharpening_vi(self):
        KERNEL = np.array([[0, 1, 0],
                           [1,-4, 1],
                           [0, 1, 0]])
        self.show_sharpening(KERNEL)
        
    def show_sharpening(self, KERNEL):
        # mengubah self.image menjadi greyscale         
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

        # ubah ke gray
        IMGGREY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # analisisi citra sebelum di convolusi
        plt.imshow(IMGGREY, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        
        # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
        hasil = self.math_konvolusi(IMGGREY, KERNEL)

        # analisis citra sesudah di konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi')

        plt.show()
        cv2.waitKey()
    
    def laplacefilter(self):
        # Load gambar
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

        # ubah ke gray
        IMGGREY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # analisisi citra sebelum di convolusi
        plt.imshow(IMGGREY, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        # buat kernel
        kernel_laplace = np.array([[0, 0, -1, 0, 0],
                                   [0, -1, -2, -1, 0],
                                   [-1, -2, 16, -2, -1],
                                   [0, -1, -2, -1, 0],
                                   [0, 0, -1, 0, 0]], dtype=np.float32)
        
        KERNEL = (1.0 / 16) * kernel_laplace
        
        # lakukan konvolusi dengan karnel dan image yang sudah di buat grey
        hasil = self.math_konvolusi(IMGGREY, KERNEL)

        # analisis citra sesudah di konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi laplace')

        plt.show()
        cv2.waitKey()
    
    def medianfilter(self):
        # mengubah self.image menjadi greyscale         
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

        # ubah ke gray
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # analisisi citra sebelum di convolusi
        plt.imshow(grey, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        # Membuat citra output dengan meng-copy citra input
        output_image = np.copy(grey)

        # Mengambil ukuran citra
        h, w = grey.shape

        # Iterasi setiap piksel pada citra (kecuali tepi citra)
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                # Membuat list untuk menyimpan nilai tetangga piksel
                neighbors = []
                # Iterasi pada tetangga piksel
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = grey[i + k, j + l]
                        neighbors.append(a)

                # Mengurutkan nilai tetangga piksel
                neighbors.sort()

                # Menempatkan nilai median pada piksel output
                median = neighbors[24]
                output_image[i, j] = copy.deepcopy(median)

        plt.figure()
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title('Median Filter')
        plt.xticks([]), plt.yticks([])

        plt.show()
        cv2.waitKey()

    def maximumfilter(self):
        # Baca citra
        img = cv2.imread("../imgs/dumy-img-1.jpg")

        # Konversi citra menjadi grayscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # menampilkan gambar asli
        plt.imshow(grey, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        # Hitung ukuran padding
        h, w = grey.shape[:2]

        # Buat temapat penampung citra keluaran dengan array 0
        img_out = np.zeros((h, w))

        # Proses Max filtering
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = grey[i + k, j + l]
                        neighbors.append(a)

                # mencari nilai max dengan nilai max
                max_val = max(neighbors)
                # membuat nilai tengahnya menjadi nilai max
                img_out.itemset((i, j), max_val)

        plt.figure()
        plt.imshow(img_out, cmap='gray')
        plt.title('Max Filtered Image')
        plt.xticks([]), plt.yticks([])

        plt.show()

    def minimumfilter(self):
        # Baca citra
        img = cv2.imread("../imgs/dumy-img-1.jpg")

        # Konversi citra menjadi grayscale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # menampilkan gambar asli
        plt.imshow(grey, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        # Hitung ukuran padding
        h, w = grey.shape[:2]

        # Buat temapat penampung citra keluaran dengan array 0
        img_out = np.zeros((h, w))

        # Proses Max filtering
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = grey[i + k, j + l]
                        neighbors.append(a)

                # mencari nilai max dengan nilai max
                max_val = min(neighbors)
                # membuat nilai tengahnya menjadi nilai max
                img_out.itemset((i, j), max_val)

        plt.figure()
        plt.imshow(img_out, cmap='gray')
        plt.title('Min Filtered Image')
        plt.xticks([]), plt.yticks([])

        plt.show()
        

    # mengatur gambar di windows
    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        
        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)

        # karna data warnanya adalah bgr bukan rgb
        img = img.rgbSwapped()

        # secara default, dia akan menampilkan gambar pada label 1
        if windows==1:
            self.label.setPixmap(QPixmap.fromImage(img))
            # memposisikan gambar
            self.label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)
        elif windows==2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)


# mempersiapkan tampilan widows
app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle("Pertemuan 2")
window.show()
sys.exit(app.exec_())

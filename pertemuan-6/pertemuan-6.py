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
        self.actionAll.triggered.connect(self.sharpening_all)   

        # Transformasi Fourier
        self.actionSmoothing.triggered.connect(self.dftsmooth)   
        self.actionEdge.triggered.connect(self.dftedge)    
        self.actionSobel.triggered.connect(self.SobelClicked)    
        self.actionPrewitt.triggered.connect(self.PrewittClicked)    
        self.actionRobert.triggered.connect(self.RobertClicked)    
        self.actionCanny_Adge.triggered.connect(self.CannyClicked)    

        # Morfologi
        self.actionDilasi.triggered.connect(self.dilasi)    
        self.actionErosi.triggered.connect(self.erosi)    
        self.actionOpening.triggered.connect(self.opening)    
        self.actionClosing.triggered.connect(self.closing)    
        self.actionSkeletonize.triggered.connect(self.skeletonize) 

        # Tresholding
        self.actionBinary.triggered.connect(self.Binary) 
        self.actionBinary_INV.triggered.connect(self.BinaryINV) 
        self.actionTrunc.triggered.connect(self.Trunc) 
        self.actionTo_Zero.triggered.connect(self.ToZero) 
        self.actionGlobalTreshold.triggered.connect(self.TugasGlobalThresh) 
        self.actionMeanTreshold.triggered.connect(self.meanThresholding) 
        self.actionGaussianTreshold.triggered.connect(self.gaussianThresholding) 
        self.actionOtsuTreshold.triggered.connect(self.otsuThresholding) 
        self.actionContour.triggered.connect(self.contour) 
           

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
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

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
        print(hasil)

        plt.show()
        cv2.waitKey()

    def konvolusi2(self):
        # mengubah self.image menjadi greyscale
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

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
        print(hasil)

        plt.show()
        cv2.waitKey()
    
    def meanfilter(self):
        # Load citra
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

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
        print(hasil2x2)

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
        print(hasil)

        plt.show()
        cv2.waitKey()
        
    def sharpening_all(self):
        self.sharpening_i()
        self.sharpening_ii()
        self.sharpening_iii()
        self.sharpening_iv()
        self.sharpening_v()
        self.sharpening_vi()

    def sharpening_i(self):
        KERNEL = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                           [-1, -1, -1]])
        self.show_sharpening(KERNEL, " i")

    def sharpening_ii(self):
        KERNEL = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        self.show_sharpening(KERNEL, " ii")

    def sharpening_iii(self):
        KERNEL = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        self.show_sharpening(KERNEL, " iii")

    def sharpening_iv(self):
        KERNEL = np.array([[1, -2, 1],
                           [-2, 5, -2],
                           [1, -2, 1]])
        self.show_sharpening(KERNEL, " iv")

    def sharpening_v(self):
        KERNEL = np.array([[1, -2, 1],
                           [-2, 4, -2],
                           [1, -2, 1]])
        self.show_sharpening(KERNEL, " v")

    def sharpening_vi(self):
        KERNEL = np.array([[0, 1, 0],
                           [1,-4, 1],
                           [0, 1, 0]])
        self.show_sharpening(KERNEL, " vi")
        
    def show_sharpening(self, KERNEL, stringkarnel):
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
        plt.title('Hasil Konvolusi' + stringkarnel)
        print(hasil)

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
        print(hasil)

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
        print(output_image)

        plt.show()
        cv2.waitKey()

    def maximumfilter(self):
        # Baca citra
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

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
        print(img_out)

        plt.show()

    def minimumfilter(self):
        # Baca citra
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

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
        print(img_out)

        plt.show()

    def dftsmooth(self):

        # Membaca gambar dari file "noisy_image.png" dan melakukan transformasi Fourier
        img = cv2.imread('../imgs/dumy-img-1.jpg', 0) # membaca image lalu ubah menjadi greyscale
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT) # mempersiapkan gambar
        dft_shift = np.fft.fftshift(dft) # karna pada dasarnya citra titik tengahnya ada di ujung kiri atas, maka harus di shift ke tengah <- menggunakan fungsi nft dari numpy
        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))) # mengambil nilai magnitude

        ## pengolahan citra dalam tingkat frekuensi ##
        # Menentukan ukuran gambar dan titik pusat
        rows, cols = img.shape
        crow, ecol = int(rows / 2), int(cols / 2)

        # Membuat low pass filter
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ecol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1 # membuat mask nya bernilai 1

        # Mengalikan filter notch dengan hasil transformasi Fourier
        fshift = dft_shift * mask # memasukan nilai mask ke dalam magnitude
        epsilon = 1e-10
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + epsilon)

        ## mengembalikan citra yang semula berbentuk spektrum menjad img kembali ##
        # Melakukan inversi transformasi Fourier
        f_ishift = np.fft.ifftshift(fshift) # mengembalikan yang sebelumnya di tengah ke ujung kiri atas
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Menampilkan gambar-gambar pada satu window
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')

        # tampilkan citra
        plt.show()

    def dftedge(self):

        # Membaca gambar dari file "noisy_image.png" dan melakukan transformasi Fourier
        img = cv2.imread('../imgs/dumy-img-1.jpg', 0) # membaca image lalu ubah menjadi greyscale
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT) # mempersiapkan gambar
        dft_shift = np.fft.fftshift(dft) # karna pada dasarnya citra titik tengahnya ada di ujung kiri atas, maka harus di shift ke tengah <- menggunakan fungsi nft dari numpy
        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))) # mengambil nilai magnitude

        ## pengolahan citra dalam tingkat frekuensi ##
        # Menentukan ukuran gambar dan titik pusat
        rows, cols = img.shape
        crow, ecol = int(rows / 2), int(cols / 2)

        # Membuat low pass filter
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 10
        center = [crow, ecol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0 # membuat mask nya bernilai 1

        # Mengalikan filter notch dengan hasil transformasi Fourier
        fshift = dft_shift * mask  # memasukan nilai mask ke dalam magnitude
        epsilon = 1e-10
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + epsilon)

        ## mengembalikan citra yang semula berbentuk spektrum menjad img kembali ##
        # Melakukan inversi transformasi Fourier
        f_ishift = np.fft.ifftshift(fshift) # mengembalikan yang sebelumnya di tengah ke ujung kiri atas
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Menampilkan gambar-gambar pada satu window
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')

        # tampilkan citra
        plt.show()
    
    def SobelClicked(self):
        # load image in grayscale mode
        img = cv2.imread('../imgs/dumy-img-1.jpg', cv2.IMREAD_GRAYSCALE)

        # tampilkan citra lama
        cv2.imshow("CItra Ori", img)

        # initialize sobel kernels
        sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # apply sobel kernels to image
        gx = cv2.filter2D(img, -1,  sobelx)
        gy = cv2.filter2D(img, -1,  sobely)
        
        # calculate gradient magnitude
        gradient = np.sqrt((gx * gx) + (gy * gy))

        # normalize gradient magnitude to 0-255 range
        gradient_norm = ((gradient / np.max(gradient)) * 255)

        # display output image
        plt.imshow(gradient_norm, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def PrewittClicked(self):
        # Baca citra
        img = cv2.imread("../imgs/salt-and-papper-img.jpg")

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        # apply sobel kernels to image
        gx = cv2.filter2D(img_gray, -1,  kernel_x)
        gy = cv2.filter2D(img_gray, -1,  kernel_y)

        # calculate gradient magnitude
        gradient = np.sqrt(gx ** 2 + gy ** 2)

        # normalize gradient magnitude to 0-255 range
        gradient_norm = (gradient * 255.0 / gradient.max()).astype(np.uint8)

        # display output image
        plt.imshow(gradient_norm, cmap='gray', interpolation='bicubic')
        cv2.imshow("ORI", img_gray)
        plt.show()

    def RobertClicked(self):
        # Baca citra
        img = cv2.imread('../imgs/dumy-img-1.jpg')

        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel_x = np.array([[0, 1], [-1, 0]])
        kernel_y = np.array([[1, 0], [0, -1]])

        # apply sobel kernels to image
        gx = cv2.filter2D(img_gray, -1, kernel_x)
        gy = cv2.filter2D(img_gray, -1, kernel_y)

        # calculate gradient magnitude
        gradient = np.sqrt(gx ** 2 + gy ** 2)
        
        # normalize gradient magnitude to 0-255 range
        gradient_norm = (gradient * 255.0 / gradient.max()).astype(np.uint8)

        # display output image
        plt.imshow(gradient_norm, cmap='gray', interpolation='bicubic')
        cv2.imshow("ORI", img_gray)
        plt.show()    

    def CannyClicked(self):
        # Load image
        img = cv2.imread("../imgs/dumy-img-1.jpg")

        plt.imshow(img[:, :, ::-1])
        img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Step 1: Noise Reduction dengan gaussian karnel
        gauss = (1.0 / 57) * np.array([[0, 1, 2, 1, 0],
                                       [1, 3, 5, 3, 1],
                                       [2, 5, 9, 5, 2],
                                       [1, 3, 5, 3, 1],
                                       [0, 1, 2, 1, 0]])
        img_out = self.math_konvolusi(img1,  gauss)
        fig = plt.figure(figsize=(12, 12))

        # Step 2: Finding Gradient
        sobel_x = cv2.Sobel(img_out, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_out, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        theta = np.arctan2(sobel_y, sobel_x)

        # Step 3: Non-Maximum Suppression
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        Z = np.zeros(img1.shape, dtype=np.int32)
        H, W = img1.shape
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = mag[i, j + 1]
                        r = mag[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = mag[i + 1, j - 1]
                        r = mag[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = mag[i + 1, j]
                        r = mag[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = mag[i - 1, j - 1]
                        r = mag[i + 1, j + 1]
                    if (mag[i, j] >= q) and (mag[i, j] >= r):
                        Z[i, j] = mag[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass
        img_N = Z.astype("uint8")

        # Step 4: Hysteresis Thresholding
        weak = 50
        strong = 150
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (weak < a < strong):  # weak
                    b = weak
                elif (a > strong):  # strong
                    b = 255
                else:
                    b = 0
                img_N.itemset((i, j), b)
        img_H1 = img_N.astype("uint8")

        # hysteresis Thresholding eliminasi titik tepi lemah jika tidak terhubung dengan tetangga tepi kuat
        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or
                                (img_H1[i + 1, j] == strong) or
                                (img_H1[i + 1, j + 1] == strong) or
                                (img_H1[i, j - 1] == strong) or
                                (img_H1[i, j + 1] == strong) or
                                (img_H1[i - 1, j - 1] == strong) or
                                (img_H1[i - 1, j] == strong) or
                                (img_H1[i - 1, j +   1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        print("error")
                        pass

        img_H2 = img_H1.astype("uint8")

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img_out, cmap='gray')
        ax1.title.set_text('Noise Reduction')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(theta, cmap='gray')
        ax2.title.set_text('Finding Gradien')
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(img_N, cmap='gray')
        ax3.title.set_text('Non-Maximum suppression')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_H2, cmap='gray')
        ax4.title.set_text('Hysterisis Thresholding')
        plt.show()

    def dilasi(this):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")
        cv2.imshow("ORI", img)

        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
        _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        # Inisialisasi Strel dengan menggunakan cv2.MORPH_CROSS, (5, 5)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        # Dilasi
        img_dilated = cv2.dilate(img_binary,strel)

        cv2.imshow('Dilated Image', img_dilated)
    
    def erosi(this):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")
        cv2.imshow("ORI", img)

        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
        _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        # Inisialisasi Strel dengan menggunakan cv2.MORPH_CROSS, (5, 5)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        # Erosi
        img_dilated = cv2.erode(img_binary,strel)

        cv2.imshow('Dilated Image', img_dilated)
    
    def opening(this):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")
        cv2.imshow("ORI", img)

        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
        _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        # Inisialisasi Strel dengan menggunakan cv2.MORPH_CROSS, (5, 5)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        
        # MORPH OPEN untuk erosi -> dilasi
        img_open = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, strel)

        cv2.imshow('Opening Image', img_open)

    def closing(this):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")
        cv2.imshow("ORI", img)

        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Mengkonversi citra grayscale menjadi citra biner menggunakan cv2.threshold
        _, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        # Inisialisasi Strel dengan menggunakan cv2.MORPH_CROSS, (5, 5)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        # MORPH CLOSE untuk dilasi -> erosi
        img_open = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, strel)

        cv2.imshow('Opening Image', img_open)

    def skeletonize(this):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")
        cv2.imshow("ORI", img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # melakukan thresholding untuk mendapatkan citra biner
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # melakukan operasi skeletonizing
        size = np.size(thresh)
        skel = np.zeros(thresh.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

        # Mengulangi proses erosi dan dilasi hingga citra tidak berubah lagi
        while True:
            erode = cv2.erode(thresh, element)
            temp = cv2.dilate(erode, element)
            temp = cv2.subtract(thresh, temp)
            skel = cv2.bitwise_or(skel, temp)
            thresh = erode.copy()

            zeros = size - cv2.countNonZero(thresh)
            if zeros == size:
                break

        cv2.imshow("Skeletonize", skel)
    
    def Binary(self):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")

        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = 127 #nilai ambang
        max = 255 #nilai derajat keabuan
        ret, thresh = cv2.threshold(img_gray,T,max, cv2.THRESH_BINARY)
        plt.figure()

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(thresh, cmap='gray')
        plt.title("Binary Image")

        plt.tight_layout()
        plt.show()
    
    def BinaryINV(self):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")

        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = 127 #nilai ambang
        max = 255 #nilai derajat keabuan
        ret, thresh = cv2.threshold(img_gray,T,max, cv2.THRESH_BINARY_INV)
        plt.figure()

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(thresh, cmap='gray')
        plt.title("Binary Invers Image")

        plt.tight_layout()
        plt.show()

    def Trunc(self):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")

        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = 127 #nilai ambang
        max = 255 #nilai derajat keabuan
        ret, thresh = cv2.threshold(img_gray,T,max, cv2.THRESH_TRUNC)
        plt.figure()

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(thresh, cmap='gray')
        plt.title("Trunc Image")

        plt.tight_layout()
        plt.show()

    def ToZero(self):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")

        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = 127 #nilai ambang
        max = 255 #nilai derajat keabuan
        ret, thresh = cv2.threshold(img_gray,T,max, cv2.THRESH_TOZERO)
        plt.figure()

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(thresh, cmap='gray')
        plt.title("To Zero Image")

        plt.tight_layout()
        plt.show()

    def ToZeroINV(self):
        # Membaca citra
        img = cv2.imread("../imgs/dumy-img-4.jpg")

        # Mengkonversi citra menjadi grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = 127 #nilai ambang
        max = 255 #nilai derajat keabuan
        ret, thresh = cv2.threshold(img_gray,T,max, cv2.THRESH_TOZERO_INV)
        plt.figure()

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(thresh, cmap='gray')
        plt.title("To Zero Invers Image")

        plt.tight_layout()
        plt.show()
    
    def TugasGlobalThresh(self):
        img_matrix = np.array([
            [3, 0, 1, 5],
            [7, 6, 0, 4],
            [2, 7, 0, 6],
            [1, 3, 5, 5]
        ], dtype=np.uint8)

        T = 4
        max_val = 7

        _, thresh_binary = cv2.threshold(img_matrix, T, max_val, cv2.THRESH_BINARY)
        _, thresh_binary_inv = cv2.threshold(img_matrix, T, max_val, cv2.THRESH_BINARY_INV)
        _, thresh_trunc = cv2.threshold(img_matrix, T, max_val, cv2.THRESH_TRUNC)
        _, thresh_to_zero = cv2.threshold(img_matrix, T, max_val, cv2.THRESH_TOZERO)
        _, thresh_to_zero_inv = cv2.threshold(img_matrix, T, max_val, cv2.THRESH_TOZERO_INV)

        print("Original Matrix:")
        print(img_matrix)
        print("\nBinary Threshold:")
        print(thresh_binary)
        print("\nBinary Inverse Threshold:")
        print(thresh_binary_inv)
        print("\nTruncated Threshold:")
        print(thresh_trunc)
        print("\nThreshold to Zero:")
        print(thresh_to_zero)
        print("\nThreshold to Zero Inverse:")
        print(thresh_to_zero_inv)   
    
    def meanThresholding(self):
        img = cv2.imread("../imgs/dumy-img-4.jpg")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("piksel awal", img)
        thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,2)
        plt.figure()

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(thresh, cmap='gray')
        plt.title(" Mean Thresholding Image")
        print("piksel thresh", thresh)
        plt.tight_layout()
        plt.show()
    
    def gaussianThresholding(self):
        img = cv2.imread("../imgs/dumy-img-4.jpg")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("piksel awal", img)
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        plt.figure()

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(thresh, cmap='gray')
        plt.title(" Gaussian Thresholding Image")
        print("piksel thresh", thresh)
        plt.tight_layout()
        plt.show()
    
    def otsuThresholding(self):
        img = cv2.imread("../imgs/dumy-img-4.jpg")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print("piksel awal", img)
        T =130
        ret ,thresh = cv2.threshold(img_gray, T,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        plt.figure()

        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        plt.subplot(122)
        plt.imshow(thresh, cmap='gray')
        plt.title(" Otsu Thresholding Image")
        print("piksel thresh", thresh)
        plt.tight_layout()
        plt.show()
    
    def contour(self):
        # Membaca citra
        img = cv2.imread('../imgs/dumy-contour-2.jpg')
        img_height, img_width, _ = img.shape
        plt.figure(figsize=(12, 6))

        # Menampilkan citra asli
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        # Mengkonversi citra menjadi grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold citra dengan nilai T=127
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Ekstrak kontur
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Loop untuk setiap kontur
        for i, cnt in enumerate(contours):
            # Mengabaikan kontur bingkai
            if cv2.contourArea(cnt) > img_width * img_height * 0.9:
                continue

            # Get approximate polygon
            epsilon = 0.01 * cv2.arcLength(cnt, True) # menghitung panjang kurva
            approx = cv2.approxPolyDP(cnt, epsilon, True) # memperhalus kurva

            # Menentukan titik tengah kontur
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])   
            cY = int(M["m01"] / M["m00"])

            # Cek bentuk berdasarkan jumlah sisi poligon
            num_sides = len(approx)
            if num_sides == 3:
                shape = "Segitiga"
            elif num_sides == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                    shape = "Persegi"
                else:
                    shape = "Persegi Panjang"
            elif num_sides == 5:
                shape = "Pentagon"
            elif num_sides == 10:
                shape = "Bintang"
            else:
                shape = "Lingkaran"

            # Menandai atau memberikan label pada setiap kontur
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)

            # Ukur teks dan atur posisi teks
            (text_width, text_height), _ = cv2.getTextSize(shape, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = cX - text_width // 2
            text_y = cY + text_height // 2

            # Menambahkan label teks pada citra
            cv2.putText(img, shape, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 122), 2)

        # Menampilkan citra dengan kontur dan label
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Image with Contours ")

        # Mengatur layout plot agar tidak saling tumpang tindih
        plt.tight_layout()

        # Menampilkan plot
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

#Import Library
import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
import numpy as np
import math
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def _init_(self):
        super(ShowImage,self)._init_()
        loadUi('A3.ui',self)
        self.image=None
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayimage)
        self.actionBrightness.triggered.connect(self.brightness)
        self.actionContrast.triggered.connect(self.Contrast)
        self.actionCliping.triggered.connect(self.cliping)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)
        self.actionHistogram_Grayscale.triggered.connect(self.grayhistogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogram)
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action_45_Derajat.triggered.connect(self.rotasi45aderajat)
        self.action45_Derajat.triggered.connect(self.rotasi45derajat)
        self.action_90_Derajat.triggered.connect(self.rotasi90aderajat)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.action180_Derajat.triggered.connect(self.rotasi180derajat)
        self.action2x.triggered.connect(self.zoomin2x)
        self.action3x.triggered.connect(self.zoomin3x)
        self.action4x.triggered.connect(self.zoomin4x)
        self.action1_2.triggered.connect(self.zoomout05)
        self.action1_4.triggered.connect(self.zoomout025)
        self.action3_4.triggered.connect(self.zoomout075)
        self.actionCrop.triggered.connect(self.crop)
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika1)
        self.actionKali_dan_Bagi.triggered.connect(self.aritmatika2)
        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        self.actionOperasi_OR.triggered.connect(self.operasiOR)
        self.actionOperasi_XOR.triggered.connect(self.operasiXOR)
        self.action1.triggered.connect(self.Konvolusi1)
        self.action2.triggered.connect(self.Konvolusi2)
        self.actionPerbandingan_1_dan_2.triggered.connect(self.Perbandingankonvolusi)
        self.action3x3.triggered.connect(self.MeanFilter3x3)
        self.action2x2.triggered.connect(self.MeanFilter2x2)
        self.actionPerbandingan_2x2_dan_3x3.triggered.connect(self.Perbandingan)
        self.actionGaussian_Filter_2.triggered.connect(self.gaussianFilter)
        self.actioni_2.triggered.connect(self.LowPassFilteri)
        self.actionii.triggered.connect(self.LowPassFilterii)
        self.actioniii.triggered.connect(self.LowPassFilteriii)
        self.actioniv.triggered.connect(self.LowPassFilteriv)
        self.actionv.triggered.connect(self.LowPassFilterv)
        self.actionvi.triggered.connect(self.LowPassFiltervi)
        self.actionLaplace.triggered.connect(self.Laplace)
        self.actionMedian.triggered.connect(self.MedianFilter)
        self.actionMax.triggered.connect(self.MaxFilter)
        self.actionMin.triggered.connect(self.MinFilter)
        self.actionDFT_Smoothing_Image.triggered.connect(self.dftsmooth)
        self.actionDFT_Edge_Image.triggered.connect(self.dftedge)
        self.actionSobel.triggered.connect(self.SobelClicked)
        self.action_Prewitt.triggered.connect(self.PrewittClicked)
        self.actionRoberts.triggered.connect(self.RobertClicked)
        self.actionCanny.triggered.connect(self.CannyClicked)
        self.actionPerbandingan_Sobel_Prewitt_Roberts.triggered.connect(self.show_sobel_prewitt_roberts)
  #modul1
    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('img.jpg')

    def loadImage(self,flname):
        self.image=cv2.imread(flname)
        self.displayImage(1)
    def grayimage(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] + 0.587* self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        self.displayImage(2)

    def brightness(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.image.itemset((i, j), b)

        self.displayImage(2)

    def Contrast(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        contras = 1.8
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = math.ceil(a * contras)

                self.image.itemset((i, j), b)
        self.displayImage(2)

    def cliping(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        contras = 1.8
        max = 255
        min = 0
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a * contras, 0, 255)

                if b > max:
                    b = max
                else:
                    if b < min:
                        b = min

                self.image.itemset((i, j), b)
        self.displayImage(2)

    def negative(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        max_intensity = 255
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(max_intensity - a, 0, 255)

                self.image.itemset((i, j), b)
        self.displayImage(2)

    def biner(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        tresshold = 128
        max = 255
        min = 0
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a, 0, 255)

                if b > tresshold:
                    b = max
                else:
                    if b < tresshold:
                        b = min

                self.image.itemset((i, j), b)
        self.displayImage(2)

    # modul2
    def grayhistogram(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] + 0.587* self.image[i, j, 1] + 0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        self.displayImage(2)
        plt.hist(self.image.ravel(), 255, [0, 255])
        plt.show()

    @pyqtSlot()
    def RGBHistogram(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.image], [i], None, [255], [0, 255])
            plt.plot(histo, color = col)
            plt.xlim([0, 255])
        self.displayImage(2)
        plt.show()

    @pyqtSlot()
    def EqualHistogram(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.image]
        self.displayImage(2)
        plt.plot(cdf_normalized, color='b')
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def translasi(self):
        h, w = self.image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.image, T, (w, h))
        self.image=img
        self.displayImage(2)
    def rotasi45aderajat(self):
        self.rotasi(-45)
    def rotasi45derajat(self):
        self.rotasi(45)
    def rotasi90aderajat(self):
        self.rotasi(-90)
    def rotasi90derajat(self):
        self.rotasi(90)
    def rotasi180derajat(self):
        self.rotasi(180)


    def rotasi(self, degree):
        h, w = self.image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2),degree, .7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.image, rotationMatrix, (h,w))
        self.image = rot_image
        self.displayImage((2))

    def zoomout05(self):
        self.zoomout(0.5)

    def zoomout025(self):
        self.zoomout(0.25)

    def zoomout075(self):
        self.zoomout(0.75)

    def zoomout(self, skala):

        resize_img = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom Out', resize_img)
        cv2.waitKey()
    def zoomin2x(self):
        self.zoomin(2)
    def zoomin3x(self):
        self.zoomin(3)
    def zoomin4x(self):
        self.zoomin(4)
    def zoomin(self,skala):
        resize_img = cv2.resize(self.image, None, fx= skala, fy= skala, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom In', resize_img)
        cv2.waitKey()

    def crop(self):
        # membaca citra
        img = cv2.imread

        # menentukan koordinat awal dan akhir
        start_row, start_col = 200, 200
        end_row, end_col = 800, 800

        # mengambil bagian citra yang diinginkan
        cropped_img = self.image[start_row:end_row, start_col:end_col]

        # menampilkan citra hasil crop
        cv2.imshow('Original', self.image)
        cv2.imshow('Crop', cropped_img)
        cv2.waitKey()

    def aritmatika1(self):
        image1 = cv2.imread('1.png', 0)
        image2 = cv2.imread('2.png', 0)
        image_tambah = image1+image2
        image_kurang = image1-image2
        cv2.imshow('image 1 Original', image1)
        cv2.imshow('image 2 Original', image2)
        cv2.imshow('image Tambah', image_tambah)
        cv2.imshow('image Kurang', image_kurang)
        cv2.waitKey()

    def aritmatika2(self):
        image1 = cv2.imread('1.png', 0)
        image2 = cv2.imread('2.png', 0)
        image_kali = image1 * image2
        image_bagi = image1 / image2
        cv2.imshow('image 1 Original', image1)
        cv2.imshow('image 2 Original', image2)
        cv2.imshow('image Kali', image_kali)
        cv2.imshow('image Bagi', image_bagi)
        cv2.waitKey()

    def operasiAND(self):
        image1 = cv2.imread('1.png', 1)
        image2 = cv2.imread('2.png', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_and(image1, image2)
        cv2.imshow('image 1 Original', image1)
        cv2.imshow('image 2 Original', image2)
        cv2.imshow('image operasi AND', operasi)
        cv2.waitKey()

    def operasiOR(self):
        image1 = cv2.imread('1.png', 1)
        image2 = cv2.imread('2.png', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_or(image1, image2)
        cv2.imshow('image 1 Original', image1)
        cv2.imshow('image 2 Original', image2)
        cv2.imshow('image operasi OR', operasi)
        cv2.waitKey()

    def operasiXOR(self):
        image1 = cv2.imread('1.png', 1)
        image2 = cv2.imread('2.png', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_xor(image1, image2)
        cv2.imshow('image 1 Original', image1)
        cv2.imshow('image 2 Original', image2)
        cv2.imshow('image operasi XOR', operasi)
        cv2.waitKey()

    def fungsi_konvolusi(self, X, F):
        # Baca ukuran citra
        H_citra, W_citra = X.shape

        # Baca ukuran kernel
        H_kernel, W_kernel = F.shape

        # Hitung nilai padding
        H = H_kernel // 2
        W = W_kernel // 2

        # Buat citra output dengan ukuran yang sama dengan citra input
        out = np.zeros((H_citra, W_citra))

        # Lakukan konvolusi
        for i in range(H + 1, H_citra - H):
            for j in range(W + 1, W_citra - W):
                # Hitung nilai konvolusi
                sum = 0
                for k in range(-H, H):
                    for l in range(-W, W):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += w * a
                out[i, j] = sum

        return out

    def Konvolusi1(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')

        # menampilkan gambar asli
        plt.subplot(1, 2, 1)
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        # mengubah gambar menjadi grayscale
        citra_masukan = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # define kernel
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

        # memanggil fungsi konvolusi
        hasil = self.fungsi_konvolusi(citra_masukan, kernel)

        # menampilkan hasil konvolusi
        plt.subplot(1, 2, 2)
        plt.imshow(hasil, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi')
        plt.show()
        cv2.waitKey()

    def Konvolusi2(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli
        plt.subplot(1, 2, 1)
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # kernel yang akan digunakan
        kernel = np.array([[6, 0, -6],
                           [6, 1, -6],
                           [6, 0, -6]])

        # melakukan konvolusi pada gambar dengan kernel
        hasil = self.fungsi_konvolusi(img1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.subplot(1, 2, 2)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi')
        plt.show()
        cv2.waitKey()

    def Perbandingankonvolusi(self):
        # membaca gambar
        img1 = cv2.imread('G.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # kernel 3x3 dan 2x2 yang akan digunakan
        kernel_1 = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])

        kernel_2 = np.array([[6, 0, -6],
                             [6, 1, -6],
                             [6, 0, -6]])

        # melakukan konvolusi pada gambar dengan kernel_1
        hasil_1 = self.fungsi_konvolusi(img1, kernel_1)

        # menampilkan gambar hasil konvolusi menggunakan kernel_1
        plt.figure()
        plt.subplot(121)
        plt.imshow(hasil_1, cmap='gray', interpolation='bicubic')
        plt.title('Kernel 1')
        plt.xticks([]), plt.yticks([])

        # melakukan konvolusi pada gambar dengan kernel_2
        hasil_2 = self.fungsi_konvolusi(img1, kernel_2)

        # menampilkan gambar hasil konvolusi menggunakan kernel_2
        plt.subplot(122)
        plt.imshow(hasil_2, cmap='gray', interpolation='bicubic')
        plt.title('Kernel 2')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def MeanFilter3x3(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli
        plt.subplot(1, 2, 1)
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # kernel yang akan digunakan
        kernel = np.array([[1 / 9, 1 / 9, 1 / 9],
                           [1 / 9, 1 / 9, 1 / 9],
                           [1 / 9, 1 / 9, 1 / 9]])

        # melakukan konvolusi pada gambar dengan kernel
        hasil = self.fungsi_konvolusi(img1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.subplot(1, 2, 2)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi')
        plt.show()
        cv2.waitKey()

    def MeanFilter2x2(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli
        plt.subplot(1, 2, 1)
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # kernel yang akan digunakan
        kernel = np.array([[1 / 4, 1 / 4],
                           [1 / 4, 1 / 4]])

        # melakukan konvolusi pada gambar dengan kernel
        hasil = cv2.filter2D(img1, -1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.subplot(1, 2, 2)
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.title('Hasil Konvolusi')
        plt.show()
        cv2.waitKey()

    def Perbandingan(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')

        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # kernel 3x3 dan 2x2 yang akan digunakan
        kernel_3x3 = np.array([[1 / 9, 1 / 9, 1 / 9],
                               [1 / 9, 1 / 9, 1 / 9],
                               [1 / 9, 1 / 9, 1 / 9]])

        kernel_2x2 = np.array([[1 / 4, 1 / 4],
                               [1 / 4, 1 / 4]])

        # melakukan konvolusi pada gambar dengan kernel 3x3
        hasil_3x3 = self.fungsi_konvolusi(img1, kernel_3x3)

        # menampilkan gambar hasil konvolusi menggunakan kernel 3x3
        plt.figure()
        plt.subplot(121)
        plt.imshow(hasil_3x3, cmap='gray', interpolation='bicubic')
        plt.title('Mean Filter 3x3')
        plt.xticks([]), plt.yticks([])

        # melakukan konvolusi pada gambar dengan kernel 2x2
        hasil_2x2 = cv2.filter2D(img1, -1, kernel_2x2)

        # menampilkan gambar hasil konvolusi menggunakan kernel 2x2
        plt.subplot(122)
        plt.imshow(hasil_2x2, cmap='gray', interpolation='bicubic')
        plt.title('Mean Filter 2x2')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def gaussianFilter(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # kernel yang akan digunakan
        kernel = (1.0 / 345) * np.array([[1, 5, 7, 5, 1],
                                         [5, 20, 33, 20, 5],
                                         [7, 33, 55, 33, 7],
                                         [5, 20, 33, 20, 5],
                                         [1, 5, 7, 5, 1]])

        hasil = self.fungsi_konvolusi(img1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.title('Gaussian Filter')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def LowPassFilteri(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Definisikan kernel filter Laplace dengan sigma
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]], dtype=np.float32)

        sigma = 1

        # Lakukan konvolusi pada gambar dengan kernel filter
        hasil = self.fungsi_konvolusi(img1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.title('Low Pass Filter')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def LowPassFilterii(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Definisikan kernel filter Laplace dengan sigma
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]], dtype=np.float32)

        sigma = 1

        # Lakukan konvolusi pada gambar dengan kernel filter
        hasil = self.fungsi_konvolusi(img1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.title('Low Pass Filter')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def LowPassFilteriii(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Definisikan kernel filter Laplace dengan sigma
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)

        sigma = 1
        # Lakukan konvolusi pada gambar dengan kernel filter
        hasil = self.fungsi_konvolusi(img1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.title('Low Pass Filter')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def LowPassFilteriv(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Definisikan kernel filter Laplace dengan sigma
        kernel = np.array([[1, -2, 1],
                           [-2, 5, -2],
                           [1, -2, 1]], dtype=np.float32)

        sigma = 1
        # Lakukan konvolusi pada gambar dengan kernel filter
        hasil = self.fungsi_konvolusi(img1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.title('Low Pass Filter')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def LowPassFilterv(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Definisikan kernel filter Laplace dengan sigma
        kernel = np.array([[1, -2, 1],
                           [-2, 4, -2],
                           [1, -2, 1]], dtype=np.float32)

        # Lakukan konvolusi pada gambar dengan kernel filter
        hasil = self.fungsi_konvolusi(img1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.title('Low Pass Filter')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def LowPassFiltervi(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Definisikan kernel filter Laplace dengan sigma
        kernel = np.array([[0, 1, 0],
                           [1, 4, 1],
                           [0, 1, 0]], dtype=np.float32)

        sigma = 0
        # Lakukan konvolusi pada gambar dengan kernel filter
        hasil = self.fungsi_konvolusi(img1, kernel)

        # menampilkan gambar hasil konvolusi
        plt.figure()
        plt.imshow(hasil, cmap='gray', interpolation='bicubic')
        plt.title('Low Pass Filter')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def Laplace(self):
        # membaca gambar
        img1 = cv2.imread('1.jpg')
        # menampilkan gambar asli sebelum dikonvolusi
        plt.imshow(img1[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # mengubah gambar menjadi grayscale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # Definisikan kernel filter Laplace
        kernel_laplace = np.array([[0, 0, -1, 0, 0],
                                   [0, -1, -2, -1, 0],
                                   [-1, -2, 16, -2, -1],
                                   [0, -1, -2, -1, 0],
                                   [0, 0, -1, 0, 0]], dtype=np.float32)

        # Normalisasi kernel filter
        kernel_laplace = (1.0 / 16) * kernel_laplace

        # Lakukan konvolusi pada citra dengan kernel filter Laplace
        laplace_img = self.fungsi_konvolusi(img1, kernel_laplace)
        # menampilkan gambar hasil konvolusi
        plt.figure()
        plt.imshow(laplace_img, cmap='gray', interpolation='bicubic')
        plt.title('Low Pass Filter')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.waitKey()

    def MedianFilter(self):
        # Baca citra
        img = cv2.imread('1.jpg')
        # menampilkan gambar asli
        plt.subplot(1, 2, 1)
        plt.imshow(img[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # Konversi citra menjadi grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Membuat citra output dengan meng-copy citra input
        output_image = np.copy(gray)

        # Mengambil ukuran citra
        h, w = gray.shape

        # Iterasi setiap piksel pada citra (kecuali tepi citra)
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                # Membuat list untuk menyimpan nilai tetangga piksel
                neighbors = []

                # Iterasi pada tetangga piksel
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = gray[i + k, j + l]
                        neighbors.append(a)

                # Mengurutkan nilai tetangga piksel
                neighbors.sort()

                # Menempatkan nilai median pada piksel output
                median = neighbors[24]
                output_image[i, j] = median

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title('Median Filter')
        plt.axis('off')
        plt.show()

    def MaxFilter(self):
        # Baca citra
        img = cv2.imread('1.jpg')
        # menampilkan gambar asli
        plt.subplot(1, 2, 1)
        plt.imshow(img[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # Konversi citra menjadi grayscale
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Hitung ukuran padding
        h, w = img.shape[:2]

        # Buat citra hasil median filtering dengan ukuran yang sama dengan citra asli
        img_out = np.zeros((h, w))

        # Proses Max filtering
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = img1[i + k, j + l]
                        neighbors.append(a)
                max_val = max(neighbors)
                img_out.itemset((i, j), max_val)

        plt.subplot(1, 2, 2)
        plt.imshow(img_out, cmap='gray')
        plt.title('Max Filtered Image')
        plt.xticks([]), plt.yticks([])

        plt.show()

    def MinFilter(self):
        # Baca citra
        img = cv2.imread('1.jpg')
        # menampilkan gambar asli
        plt.subplot(1, 2, 1)
        plt.imshow(img[:, :, ::-1])
        plt.xticks([]), plt.yticks([])
        plt.title('Citra Asli')
        # Konversi citra menjadi grayscale
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Hitung ukuran padding

        h, w = img.shape[:2]

        # Buat citra hasil median filtering dengan ukuran yang sama dengan citra asli
        img_out = np.zeros((h, w))

        # Proses Max filtering
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                neighbors = []
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        a = img1[i + k, j + l]
                        neighbors.append(a)
                min_val = min(neighbors)
                img_out.itemset((i, j), min_val)

        plt.subplot(1, 2, 2)
        plt.imshow(img_out, cmap='gray')
        plt.title('Min Filtered Image')
        plt.xticks([]), plt.yticks([])

        plt.show()

    def dftsmooth(self):

        # Menghasilkan sinyal gelombang sinus dengan frekuensi 3 dan menambahkan noise
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)
        y += max(y)

        # Membuat gambar menggunakan sinyal gelombang sinus
        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)
        plt.imshow(img)

        # Membaca gambar dari file "noisy_image.png" dan melakukan transformasi Fourier
        img = cv2.imread('1.jpg', 0)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

        # Menentukan ukuran gambar dan titik pusat
        rows, cols = img.shape
        crow, ecol = int(rows / 2), int(cols / 2)

        # Membuat filter notch
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ecol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1

        # Mengalikan filter notch dengan hasil transformasi Fourier
        fshift = dft_shift * mask
        epsilon = 1e-10
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + epsilon)

        # Melakukan inversi transformasi Fourier
        f_ishift = np.fft.ifftshift(fshift)
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
        plt.show()

    def dftedge(self):
        # Membuat gambar grayscale dengan pola sinusoidal dan menampilkannya
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)
        y += max(y)
        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)
        plt.imshow(img)

        # Membaca gambar dari file, melakukan transformasi Fourier diskrit, dan menggeser hasilnya agar frekuensi nol berada di tengah gambar
        img = cv2.imread('1.jpg', 0)
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Menghitung magnitude spectrum dan menampilkan gambar hasil FFT
        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')

        # Membuat mask lingkaran dan mengalikannya dengan hasil shift Fourier
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 80  
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 1
        fshift = dft_shift * mask

        # Menghitung magnitude spectrum hasil pengalikan dengan mask lingkaran dan menampilkan gambar
        epsilon = 1e-8
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + epsilon)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')

        # Menggeser hasil shift Fourier yang telah dikalikan dengan mask lingkaran dan melakukan transformasi Fourier invers untuk mengembalikan gambar ke domain spasial
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # Menampilkan gambar hasil invers Fourier
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')
        plt.show()



    def SobelClicked(self):
        # load image in grayscale mode
        img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)

        # initialize sobel kernels
        sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # apply sobel kernels to image
        gx = self.fungsi_konvolusi(img,  sobelx)
        gy = self.fungsi_konvolusi(img,  sobely)

        # calculate gradient magnitude
        gradient = np.sqrt(gx ** 2 + gy ** 2)
        # normalize gradient magnitude to 0-255 range
        gradient_norm = (gradient * 255.0 / gradient.max()).astype(np.uint8)

        # display output image
        plt.imshow(gradient_norm, cmap='gray', interpolation='bicubic')
        plt.show()

    def PrewittClicked(self):
        # Baca citra
        img = cv2.imread('1.jpg')
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        # apply sobel kernels to image
        gx = self.fungsi_konvolusi(img_gray,  kernel_x)
        gy = self.fungsi_konvolusi(img_gray,  kernel_y)

        # calculate gradient magnitude
        gradient = np.sqrt(gx ** 2 + gy ** 2)
        # normalize gradient magnitude to 0-255 range
        gradient_norm = (gradient * 255.0 / gradient.max()).astype(np.uint8)

        # display output image
        plt.imshow(gradient_norm, cmap='gray', interpolation='bicubic')
        plt.show()

    def RobertClicked(self):
        # Baca citra
        img = cv2.imread('1.jpg')
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kernel_x = np.array([[0, 1], [-1, 0]])
        kernel_y = np.array([[1, 0], [0, -1]])

        # apply sobel kernels to image
        gx = cv2.filter2D(img_gray,  kernel_x)
        gy = cv2.filter2D(img_gray, -1, kernel_y)

        # calculate gradient magnitude
        gradient = np.sqrt(gx ** 2 + gy ** 2)
        # normalize gradient magnitude to 0-255 range
        gradient_norm = (gradient * 255.0 / gradient.max()).astype(np.uint8)

        # display output image
        plt.imshow(gradient_norm, cmap='gray', interpolation='bicubic')
        plt.show()
        
    def show_sobel_prewitt_roberts(self):
        img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Original')
        # initialize sobel kernels
        sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # apply sobel kernels to image
        gx_sobel = self.fungsi_konvolusi(img,  sobelx)
        gy_sobel = self.fungsi_konvolusi(img,  sobely)
        # calculate gradient magnitude for sobel
        gradient_sobel = np.sqrt(gx_sobel ** 2 + gy_sobel ** 2)
        # normalize gradient magnitude to 0-255 range
        gradient_norm_sobel = (gradient_sobel * 255.0 / gradient_sobel.max()).astype(np.uint8)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(gradient_norm_sobel, cmap='gray', interpolation='bicubic')
        ax2.title.set_text('Sobel')
        img1 = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
        # initialize prewitt kernels
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        # apply prewitt kernels to image
        gx_prewitt = self.fungsi_konvolusi(img1,  kernel_x)
        gy_prewitt = self.fungsi_konvolusi(img1,  kernel_y)
        # calculate gradient magnitude for prewitt
        gradient_prewitt = np.sqrt(gx_prewitt ** 2 + gy_prewitt ** 2)
        # normalize gradient magnitude to 0-255 range
        gradient_norm_prewitt = (gradient_prewitt * 255.0 / gradient_prewitt.max()).astype(np.uint8)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(gradient_norm_prewitt, cmap='gray', interpolation='bicubic')
        ax3.title.set_text('Prewitt')
        img2 = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
        # initialize roberts kernels
        kernel_xr = np.array([[0, 1], [-1, 0]])
        kernel_yr = np.array([[1, 0], [0, -1]])
        # apply roberts kernels to image
        gx_roberts = cv2.filter2D(img2, -1, kernel_xr)
        gy_roberts = cv2.filter2D(img2, -1, kernel_yr)
        # calculate gradient magnitude for roberts
        gradient_roberts = np.sqrt(gx_roberts ** 2 + gy_roberts ** 2)
        # normalize gradient magnitude to 0-255 range
        gradient_norm_roberts = (gradient_roberts * 255.0 / gradient_roberts.max()).astype(np.uint8)
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(gradient_norm_roberts, cmap='gray', interpolation='bicubic')
        ax4.title.set_text('Roberts')
    def CannyClicked(self):

        # Load image
        img = cv2.imread("1.jpg")
        plt.imshow(img[:, :, ::-1])
        img1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Step 1: Noise Reduction
        gauss = (1.0 / 57) * np.array([[0, 1, 2, 1, 0],
                                       [1, 3, 5, 3, 1],
                                       [2, 5, 9, 5, 2],
                                       [1, 3, 5, 3, 1],
                                       [0, 1, 2, 1, 0]])
        img_out = self.fungsi_konvolusi(img1,  gauss)
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img_out, cmap='gray')
        ax1.title.set_text('Noise Reduction')
        # Step 2: Finding Gradient
        sobel_x = cv2.Sobel(img_out, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_out, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        theta = np.arctan2(sobel_y, sobel_x)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(theta, cmap='gray')
        ax2.title.set_text('Finding Gradien')
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
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(img_N, cmap='gray')
        ax3.title.set_text('Non-Maximum suppression')
        # Step 4: Hysteresis Thresholding
        weak = 100
        strong = 150
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):  # weak
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
                                (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass
        img_H2 = img_H1.astype("uint8")
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_H2, cmap='gray')
        ax4.title.set_text('Hysterisis Thresholding')
        plt.show()
    def displayImage(self, windows):
        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:  # row[0],col[1],channel[2]
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()
        if windows == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)
        if windows == 2:
            self.hasilLabel.setPixmap(QPixmap.fromImage(img))
            self.hasilLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.hasilLabel.setScaledContents(True)

app=QtWidgets.QApplication(sys.argv)
window=ShowImage()
window.setWindowTitle('Pertemuan 5')
window.show()
sys.exit(app.exec_())

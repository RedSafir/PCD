import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
from networkx import degree



class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI_P9.ui', self)
        self.Image = None
        self.butto_loadCitra.clicked.connect(self.fungsi)
        self.butto_prosesCitra.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Con.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStreching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogram)
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action45_Derajat.triggered.connect(self.rotasi45d)
        self.action_45_Derajat.triggered.connect(self.rotasimin45d)
        self.action90_Derajat.triggered.connect(self.rotasi90d)
        self.action_90_Derajat.triggered.connect(self.rotasimin90d)
        self.action180_Deraajt.triggered.connect(self.rotasi180d)
        self.action2X.triggered.connect(self.ZoomIn2)
        self.action3X.triggered.connect(self.ZoomIn3)
        self.action4x.triggered.connect(self.ZoomIn4)
        self.action1_2.triggered.connect(self.ZoomOut1per2)
        self.action1_4.triggered.connect(self.ZoomOut1per4)
        self.action3_4.triggered.connect(self.ZoomOut3per4)
        self.actionCrop.triggered.connect(self.crop)
        self.actionTambah_dan_Kurang.triggered.connect(self.tambahkurang)
        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        self.actionOperasi_OR.triggered.connect(self.operasiOR)
        self.actionOperasi_XOR.triggered.connect(self.operasiXOR)
        self.actionKali_dan_Bagi.triggered.connect(self.kalibagi)
        self.actionKonvolusi_A.triggered.connect(self.FilteringCliked)
        self.actionKonvolusi_B.triggered.connect(self.Filterring2)
        self.actionKernel_1_4.triggered.connect(self.Mean2x2)
        self.actionKernel_1_9.triggered.connect(self.Mean3x3)
        self.actionGaussian_Filter.triggered.connect(self.Gaussian)
        self.actionKe_1.triggered.connect(self.Sharpening1)
        self.actionKe_2.triggered.connect(self.Sharpening2)
        self.actionKe_3.triggered.connect(self.Sharpening3)
        self.actionKe_4.triggered.connect(self.Sharpening4)
        self.actionKe_5.triggered.connect(self.Sharpening5)
        self.actionKe_6.triggered.connect(self.Sharpening6)
        self.actionLaplace.triggered.connect(self.Laplace)
        self.actionMedian_Filter.triggered.connect(self.Median)
        self.actionMaxFilter.triggered.connect(self.Max)
        self.actionMinFilter.triggered.connect(self.Min)
        self.actionDFT_Smoothing_Image.triggered.connect(self.SmoothImage)
        self.actionDFT_Edge_Detection.triggered.connect(self.EdgeDetec)
        self.actionOperasi_Sobel.triggered.connect(self.Opsobel)
        self.actionOperasi_Prewitt.triggered.connect(self.Opprewitt)
        self.actionOperasi_Robert.triggered.connect(self.Oprobert)
        self.actionOperasi_Canny.triggered.connect(self.OpCanny)
        self.actionErosi.triggered.connect(self.Erosi)
        self.actionDilasi.triggered.connect(self.Dilasi)
        self.actionOpening.triggered.connect(self.Opening)
        self.actionClosing.triggered.connect(self.Closing)
        self.actionBinary_2.triggered.connect(self.ThresholdingBinary)
        self.actionInverse_Binary.triggered.connect(self.ThresholdingInverseBinary)
        self.actionTrunc_2.triggered.connect(self.ThresholdingTrunc)
        self.actionTo_Zero_2.triggered.connect(self.ThresholdingToZero)
        self.actionInverse_To_Zero_2.triggered.connect(self.ThresholdingInverseToZero)
        self.actionMean_Thresholding.triggered.connect(self.MeanThresholding)
        self.actionGaussian_Thresholding.triggered.connect(self.GaussianThresholding)
        self.actionOtsu_Thresholding.triggered.connect(self.OtsuThresholding)
        self.actionContour.triggered.connect(self.Contour)
        self.actionSkeletonizing.triggered.connect(self.Skeletonizing)
        self.actionColor_Tracking.triggered.connect(self.ColorTracking)
        self.actionColor_Picker.triggered.connect(self.ColorPicker)
        self.actionObject_detection.triggered.connect(self.Objectdetection)
        self.actionHistogram_of_Gradient.triggered.connect(self.HistogramofGradient)
        self.actionFace_Detection.triggered.connect(self.FaceDetection)
        self.actionPedestrian.triggered.connect(self.Pedestrian)
        self.actionCircle_Hough.triggered.connect(self.CircleHough)




    def fungsi(self):
        self.Image = cv2.imread('lambung.jpg')
        self.displayImage(1)

    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)

    def brightness(self):

        try:
            self.Image = cv2.cvColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a =self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.Image.itemset((i, j), b)

        self.displayImage(1)

    def contrast(self):
        try:
            self.Image = cv2.cvColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i, j), b)

        self.displayImage(1)

    def contrastStreching(self):
        try:
            self.Image = cv2.cvColor(self.Image, cv2.COLOR_BGR2GRAY)

        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)

        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.Image.itemset((i, j), b)

        self.displayImage(1)

    def negative(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                a = np.clip(255 - self.Image[i, j, 0], 0, 255)
                gray.itemset((i, j), a)

        self.Image = gray
        print(self.Image)
        self.displayImage(2)

    def biner(self):
        img =cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        H, W = self.Image.shape[:2]
        for i in np.arange(H):
            for j in np.arange(W):
                a = img.item(i, j)
                if a == 180 :
                    a = 0
                elif a < 180 :
                    a = 1
                else :
                    a = 255

        self.Image = img
        print(self.Image)
        self.displayImage(2)

    def grayHistogram(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(2)
        plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()

    def RGBHistogram(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.Image], [i], None, [256], [0, 256])
            plt.plot(histo, color=col)
            plt.xlim([0, 256])
        self.displayImage(2)
        plt.show()

    def EqualHistogram(self):
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.Image = cdf[self.Image]
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.Image.flatten(), 256, [0, 256], color="r")
        plt.xlim([0, 256])
        plt.legend(("cdf", "histogram"), loc="upper left")
        plt.show()

    def translasi(self):
        H, W = self.Image.shape[:2]
        quarter_H, quarter_W = H/4, W/4
        T = np.float32([[1, 0, quarter_W], [0, 1, quarter_H]])
        img = cv2.warpAffine(self.Image, T, (W, H))
        self.Image = img
        self.displayImage(2)

    def rotasi90d(self):
        self.rotasi(90)

    def rotasi45d(self):
        self.rotasi(45)

    def rotasimin45d(self):
        self.rotasi(-45)

    def rotasimin90d(self):
        self.rotasi(-90)

    def rotasi180d(self):
        self.rotasi(180)

    def rotasi(self, degree):
        H, W = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((W / 2, H / 2), degree, .7)

        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((H * sin) + (W * cos))
        nH = int((H * cos) + (W * sin))

        rotationMatrix[0, 2] += (nW / 2) - W / 2
        rotationMatrix[1, 2] += (nH / 2) - H / 2

        rot_Image = cv2.warpAffine(self.Image, rotationMatrix, (H, W))
        self.Image = rot_Image
        self.displayImage(2)

    def ZoomIn2(self):
        skala = 2
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_Image)
        cv2.waitKey()

    def ZoomIn3(self):
        skala = 3
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_Image)
        cv2.waitKey()

    def ZoomIn4(self):
        skala = 4
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_Image)
        cv2.waitKey()


    def ZoomOut1per2(self):
        skala = 0.5
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom Out', resize_Image)

        cv2.waitKey()

    def ZoomOut1per4(self):
        skala = 0.25
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom Out', resize_Image)

        cv2.waitKey()

    def ZoomOut3per4(self):
        skala = 0.75
        resize_Image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom Out', resize_Image)

        cv2.waitKey()


    def crop(self):
        H, W = self.Image.shape[:2]
        # get the strating point of pixel coord(top left)
        start_row, start_col = int(H * .1), int(W * .1)
        # get the ending point coord (botoom right)
        end_row, end_col = int(H * .5), int(W * .5)
        crop = self.Image[start_row:end_row, start_col:end_col]

        cv2.imshow('Original', self.Image)
        cv2.imshow('Crop Image', crop)

    def tambahkurang(self):
        image1 = cv2.imread('lambung.jpg', 0)
        image2 = cv2.imread('organ.jpg', 0)
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Tambah', image_tambah)
        cv2.imshow('Image Kurang', image_kurang)
        cv2.waitKey()

    def kalibagi(self):
        image1 = cv2.imread('lambung.jpg', 0)
        image2 = cv2.imread('organ.jpg', 0)
        image_kali = image1 * image2
        image_bagi = image1 / image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Kali', image_kali)
        cv2.imshow('Image Bagi', image_bagi)
        cv2.waitKey()

    def operasiAND(self):
        image1 = cv2.imread('lambung.jpg', 0)
        image2 = cv2.imread('organ.jpg', 0)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_and(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi AND', operasi)

        cv2.waitKey()

    def operasiOR(self):
        image1 = cv2.imread('lambung.jpg', 0)
        image2 = cv2.imread('organ.jpg', 0)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_or(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi 0R', operasi)

        cv2.waitKey()

    def operasiXOR(self):
        image1 = cv2.imread('lambung.jpg', 0)
        image2 = cv2.imread('organ.jpg', 0)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_xor(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi XOR', operasi)

        cv2.waitKey()

    def Konvolusi(self, X, F):
        X_height = X.shape[0]
        X_width = X.shape[1]

        F_height = F.shape[0]
        F_width = F.shape[1]

        H = (F_height) // 2
        W = (F_width) // 2

        out = np.zeros((X_height, X_width))

        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum
        return out

    def FilteringCliked(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        kernel = np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ])

        img_out = self.Konvolusi(img, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Filterring2(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        kernel = np.array(
            [
                [6, 0, -6],
                [6, 1, -6],
                [6, 0, -6]
            ]
        )

        img_out = self.Konvolusi(img, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Mean2x2(self):
        mean = (1.0 / 4) * np.array(
            [
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 1]
            ]
        )
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        img_out = self.Konvolusi(img, mean)
        print('---Nilai Pixel Mean Filter 2x2 ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Mean3x3(self):
        mean = (1.0 / 9) * np.array(
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]
        )
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        img_out = self.Konvolusi(img, mean)
        print('---Nilai Pixel Mean Filter 3x3---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Gaussian(self):
        gausian = (1.0 / 345) * np.array(
            [
                [1, 5, 7, 5, 1],
                [5, 20, 33, 20, 5],
                [7, 33, 55, 33, 7],
                [5, 20, 33, 20, 5],
                [1, 5, 7, 5, 1]
            ]
        )
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        img_out = self.Konvolusi(img, gausian)
        print('---Nilai Pixel Gaussian ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Sharpening1(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Kernel i ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening2(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]
            ])
        img_out = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Kernel ii ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening3(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel iii ---\n', img_out)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening4(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [1, -2, 1],
                [-2, 5, -2],
                [1, 2, 1]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel iv ---\n', img_out)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening5(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [1, -2, 1],
                [-2, 4, -2],
                [1, -2, 1]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel v ---\n', img_out)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening6(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel vi ---\n', img_out)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Laplace(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        sharpe = (1.0 / 16) * np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0]
            ])
        img_out = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Median(self):  # D5
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                neighbors = []
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        neighbors.append(a)
                neighbors.sort()
                median = neighbors[24]
                b = median
                img_out.itemset((i, j), b)
        print('---Nilai Pixel Median Filter---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Max(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                max = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                        b = max
                img_out.itemset((i, j), b)
        print('---Nilai Pixel Maximun Filter ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Min(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_RGB2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                min = 0
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)
                        if a < min:
                            min = a
                        b = min
                img_out.itemset((i, j), b)
        print('---Nilai Pixel Minimun Filter ---\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def SmoothImage(self):
            x = np.arange(256)
            y = np.sin(2 * np.pi * x / 3)
            y += max(y)
            img = np.array([[y[j] * 127 for j in range(256)] for i in
                            range(256)], dtype=np.uint8)

            plt.imshow(img)
            img = cv2.imread('lbnoise.jpg', 0)
            dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols, 2), np.uint8)
            r = 90
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) * 2 + (y - center[1]) * 2 <= r * r
            mask[mask_area] = 1

            fshift = dft_shift * mask
            fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
            f_ishift = np.fft.ifftshift(fshift)

            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

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

    def EdgeDetec(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)

        y += max(y)

        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

        plt.imshow(img)
        img = cv2.imread("lbnoise.jpg", 0)

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 90
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0] * 2 + (y - center[1])) * 2 <= r * r

        mask[mask_area] = 0

        fshift = dft_shift * mask
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        f_ishift = np.fft.ifftshift(fshift)

        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

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
        ax4.title.set_text('Inverse fourier')
        plt.show()



    def Opsobel(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        X = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]])
        Y = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
        img_Gx = self.Konvolusi(img, X)
        img_Gy = self.Konvolusi(img, Y)
        img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
        img_out = (img_out / np.max(img_out)) * 255
        print('---Nilai Pixel Operasi Sobel--- \n', img_out)
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def Opprewitt(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        prewit_X = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])
        prewit_Y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]])
        img_Gx = self.Konvolusi(img, prewit_X)
        img_Gy = self.Konvolusi(img, prewit_Y)
        img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
        img_out = (img_out / np.max(img_out)) * 255
        print('---Nilai Pixel Operasi Prewitt --- \n', img_out)
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def Oprobert(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        RX = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 0]])
        RY = np.array([[0, 1, 0],
                       [-1, 0, 0],
                       [0, 0, 0]])
        img_Gx = self.Konvolusi(img, RX)
        img_Gy = self.Konvolusi(img, RY)
        img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
        img_out = (img_out / np.max(img_out)) * 255
        print('---Nilai Pixel Operasi Robert--- \n', img_out)
        self.Image = img
        self.displayImage(2)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()

    def OpCanny(self):

        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        gaus = (1.0 / 57) * np.array(
            [[0, 1, 2, 1, 0],
             [1, 3, 5, 3, 1],
             [2, 5, 9, 5, 2],
             [1, 3, 5, 3, 1],
             [0, 1, 2, 1, 0]])
        img_out = self.Konvolusi(img, gaus)
        img_out = img_out.astype("uint8")
        cv2.imshow("Noise Reduction", img_out)

        Gx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Gy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        konvolusi_x = self.Konvolusi(img, Gx)
        konvolusi_y = self.Konvolusi(img, Gy)

        theta = np.arctan2(konvolusi_y, konvolusi_x)
        theta = theta.astype("uint8")
        cv2.imshow("Finding Gradien", theta)


        H, W = img.shape[:2]
        Z = np.zeros((H, W), dtype=np.int32)

        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255

                    # Angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img_out[i, j + 1]
                        r = img_out[i, j - 1]
                    # Angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    # Angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                    # Angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                        Z[i, j] = img_out[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        img_N = Z.astype("uint8")
        cv2.imshow("Non Maximum Supression", img_N)


        weak = 80
        strong = 110
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):
                    b = weak
                    if (a > strong):
                        b = 255
                else:
                    b = 0
                img_N.itemset((i, j), b)
        img_H1 = img_N.astype("uint8")
        cv2.imshow("Hysterisis part 1", img_H1)
        print('---Nilai Pixel Hysterisis Part 1--- \n', img_H1)


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
        cv2.imshow("Hysteresis part 2", img_H2)
        print('---Nilai Pixel Hysterisis Part 2--- \n', img_H2)


    def Erosi(self):
        
        img = cv2.imread('lambung.jpg', 0)

        ret, biner = cv2.threshold(img, 127, 255, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        erosi = cv2.erode(biner, kernel)

        cv2.imshow('Erosi', erosi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Dilasi(self):

        img = cv2.imread('lambung.jpg', 0)

        ret, biner = cv2.threshold(img, 127, 255, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        img_dilated = cv2.dilate(biner, kernel)

        cv2.imshow('Citra Hasil Dilasi', img_dilated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Opening(self):

        img = cv2.imread('lambung.jpg', 0)

        ret, img_bin = cv2.threshold(img, 127, 255, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        img_opening = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel)

        cv2.imshow('Opening', img_opening)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def Closing(self):

        img = cv2.imread('lambung.jpg', cv2.IMREAD_GRAYSCALE)

        ret, biner = cv2.threshold(img, 127, 255, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        closed_img = cv2.morphologyEx(biner, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('Closing', closed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def Skeletonizing(self):

        img = cv2.imread('organ.jpg', 0)
        size = np.size(img)

        skel = np.zeros(img.shape, np.uint8)

        ret, img = cv2.threshold(img, 127, 255, 0)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        done = False
        while (not done):
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros == size:
                done = True

        cv2.imshow("skel", skel)
        cv2.waitKey(0)

        print("Original", img)
        print("Skeletonizing", skel)

        cv2.destroyAllWindows()


    def ThresholdingBinary(self):

        img_color = cv2.imread('lambung.jpg')

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        threshold = 127
        max_value = 255

        ret, img_binary = cv2.threshold(img_gray, threshold, max_value, cv2.THRESH_BINARY)

        cv2.imshow('Binary Thresholding Image', img_binary)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def ThresholdingInverseBinary(self):

        img_color = cv2.imread('lambung.jpg')

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        threshold = 127
        max_value = 255

        ret, img_binary_inv = cv2.threshold(img_gray, threshold, max_value, cv2.THRESH_BINARY_INV)

        cv2.imshow('Inverse Binary Thresholding Image', img_binary_inv)
        cv2.waitKey(0)

        cv2.destroyAllWindows()

    def ThresholdingTrunc(self):

        img_color = cv2.imread('lambung.jpg')

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        threshold = 200
        max_value = 255

        ret, img_trunc = cv2.threshold(img_gray, threshold, max_value, cv2.THRESH_TRUNC)

        cv2.imshow('Trunc Thresholding Image', img_trunc)
        cv2.waitKey(0)

        cv2.destroyAllWindows()


    def ThresholdingToZero(self):

        img_color = cv2.imread('lambung.jpg')

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        threshold = 127
        max_value = 255

        ret, img_tozero = cv2.threshold(img_gray, threshold, max_value, cv2.THRESH_TOZERO)


        cv2.imshow('To Zero Thresholding Image', img_tozero)
        cv2.waitKey(0)


        cv2.destroyAllWindows()


    def ThresholdingInverseToZero(self):

        img = cv2.imread("lambung.jpg")


        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        threshold_value = 100
        max_value = 255


        ret, thresh_img = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_TOZERO_INV)


        cv2.imshow("Inverse To ZeroImage", thresh_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def MeanThresholding(self):
        img = cv2.imread("lambung.jpg", 0)

        print("Piksel Citra Awal", img)

        image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)

        cv2.imshow("Mean Thresholding", image)
        print("Piksel Mean Thresholding", image)


    def GaussianThresholding(self):
        img = cv2.imread("lambung.jpg", 0)

        image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)

        cv2.imshow("Gaussian Thresholding", image)
        print("Piksel Gaussian Thresholding", image)


    def OtsuThresholding(self):
        img = cv2.imread("lambung.jpg", 0)

        T = 130

        ret, img_otsu = cv2.threshold(img, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        cv2.imshow("Otsu Thresholding", img_otsu)
        print("Piksel Otsu Thresholding", img_otsu)


    def Contour(self):
        path, _ = QFileDialog.getOpenFileName()
        Img = cv2.imread(path)
        gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i = 0
        for contour in contours:
            if i == 0:
                i = 1
                continue
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            cv2.drawContours(Img, [contour], 0, (0, 0, 255), 5)
            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
            if len(approx) == 3:
                cv2.putText(Img, 'Triangle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            elif len(approx) == 4:
                cv2.putText(Img, 'Square', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            elif len(approx) == 5:
                cv2.putText(Img, 'Pentagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            elif len(approx) == 6:
                cv2.putText(Img, 'Hexagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            elif len(approx) == 8:
                cv2.putText(Img, 'Octagon', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            elif len(approx) == 10:
                cv2.putText(Img, 'Star', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
            else:
                cv2.putText(Img, 'circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        cv2.imshow('Contour', Img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def ColorTracking(self):
        cam = cv2.VideoCapture(0)
        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_color = np.array([20, 60, 75])
            upper_color = np.array([110, 255, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            result = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)
            key = cv2.waitKey(1)
            if key == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
        print(result)


    def ColorPicker(self):
        # Fungsi untuk trackbar
        def nothing(x):
            pass

        # Inisialisasi webcam
        cap = cv2.VideoCapture(0)

        # Membuat trackbar untuk nilai HSV
        cv2.namedWindow('Tracking')
        cv2.createTrackbar('LH', 'Tracking', 0, 179, nothing)
        cv2.createTrackbar('LS', 'Tracking', 0, 255, nothing)
        cv2.createTrackbar('LV', 'Tracking', 0, 255, nothing)
        cv2.createTrackbar('UH', 'Tracking', 179, 179, nothing)
        cv2.createTrackbar('US', 'Tracking', 255, 255, nothing)
        cv2.createTrackbar('UV', 'Tracking', 255, 255, nothing)

        while True:
            # Membaca frame dari webcam
            _, frame = cap.read()

            # Konversi frame dari RGB ke HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Ambil nilai dari trackbar
            lh = cv2.getTrackbarPos('LH', 'Tracking')
            ls = cv2.getTrackbarPos('LS', 'Tracking')
            lv = cv2.getTrackbarPos('LV', 'Tracking')
            uh = cv2.getTrackbarPos('UH', 'Tracking')
            us = cv2.getTrackbarPos('US', 'Tracking')
            uv = cv2.getTrackbarPos('UV', 'Tracking')

            # Batas bawah dan batas atas warna
            lower_color = np.array([lh, ls, lv])
            upper_color = np.array([uh, us, uv])

            # Membuat masking dari nilai batas bawah dan batas atas
            mask = cv2.inRange(hsv, lower_color, upper_color)

            # Membuat hasil color tracking
            res = cv2.bitwise_and(frame, frame, mask=mask)

            # Menampilkan hasil
            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask)
            cv2.imshow('Result', res)

            # Tombol untuk keluar dari loop
            key = cv2.waitKey(1)
            if key == 27:
                break

        # Menutup webcam dan semua window
        cap.release()
        cv2.destroyAllWindows()

    def Objectdetection(self):
        cam = cv2.VideoCapture('cars.mp4')
        car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

        while True:
            ret, frame = cam.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cars = car_cascade.detectMultiScale(gray, 1.1, 3)

            for (x, y, w, h) in cars:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            cv2.imshow('traffic', frame)
            if cv2.waitKey(10) == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()


    def HistogramofGradient(self):
        import cv2
        import imutils
        from skimage.feature import hog
        from skimage import data, exposure
        import matplotlib.pyplot as plt
        from skimage.io import imread
        from skimage.transform import resize
        from skimage import exposure

        img = data.astronaut()
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        photo = cv2.imread("pedestrian.jpg")

        photo = imutils.resize(photo, width=min(400, photo.shape[1]))

        (regions, _) = hog.detectMultiScale(photo, winStride=(4, 4), padding=(4, 4), scale=1.05)

        for (x, y, w, h) in regions:
            cv2.rectangle(photo, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("image", photo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def FaceDetection(self):
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        Image = cv2.imread('face.jpg')
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if faces == ():
            print("No faces found")
        for (x, y, w, h) in faces:
            cv2.rectangle(Image, (x, y), (x + w, y + h), (127, 0, 255), 2)
            cv2.imshow('Face Detection', Image)
            cv2.waitKey(0)


    def Pedestrian(self):
        body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')
        # Initiate video capture for video file
        cap = cv2.VideoCapture('people.mp4')
        # Loop once video is successfully loaded
        while cap.isOpened():
            # Read first frame
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                            interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Pass frame to our body classifier
            bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
            # Extract bounding boxes for any bodies identified
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.imshow('Pedestrians', frame)
            if cv2.waitKey(1) == 13:  # 13 is the Enter Key
                break
        cap.release()
        cv2.destroyAllWindows()

    def CircleHough(self):
        img = cv2.imread('cv.png', 0)
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                                   param1=50, param2=30, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imshow('detected circles', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def displayImage(self, window):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        if window == 1:
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setScaledContents(True)

        if window == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))
            self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label_2.setScaledContents(True)



app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 4')
window.show()
sys.exit(app.exec_())


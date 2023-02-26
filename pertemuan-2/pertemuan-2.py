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

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("GUI.ui", self)
        self.Image = None
        # akan membaca bila action di tekan
        self.button_loadCitra.clicked.connect(self.fungsi)
        self.button_prosesCitra.clicked.connect(self.greyscale)
        self.actionOperasi_pencerahan.triggered.connect(self.brightness)
        self.actionSimple_contras.triggered.connect(self.contras)
        self.actionContras_streching.triggered.connect(self.contrasstreching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)
        self.actionHistogram_greyscale.triggered.connect(self.greyHistogram)
        self.actionHistogram_RGB.triggered.connect(self.rgbHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.equalzationHistogram)
        self.actionTranslasi.triggered.connect(self.translasi)

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
        BRIGHTNES = 80
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
        CONTRAS = 1.7
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
            histo = cv2.calcHist([self.Image], [i], None, [256], [0,256])
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

    def rotasi(self, degree):
        h, w = self.Image.shape[:2] # mendapatkan bentuk width dan height nya gambar
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, .7) # jari-jari di kali derajat

        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (h, w))

        self.Image = rot_image

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
window.setWindowTitle("Pertemuan 1")
window.show()
sys.exit(app.exec_())

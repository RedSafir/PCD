#Import Library
import cv2
import sys
from PyQt5 import QtCore,QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
from PyQt5.uic import loadUi
import numpy as np

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage,self).__init__()
        loadUi('A3.ui',self)
        self.image=None
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayimage)
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
window.setWindowTitle('Show Image GUI')
window.show()
sys.exit(app.exec_())
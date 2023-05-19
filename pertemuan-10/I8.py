# untuk mendeteksi landmark pada wajah

import cv2
import dlib #untuk deteksi wajah dan landmark biasanya digunakan untuk pengenalan pola
import numpy 



PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat" #menentukan PATH untuk file model landmark wajah yang akan digunakan pada variabel PREDICTOR_PATH
#membuat objek predictor dan detector, yang akan digunakan untuk mendeteksi wajah dan landmark pada gambar
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector() # adalh sebuah object detector dari library

#mendefinisikan 2 exception
class TooManyFaces(Exception):
    pass
class NoFaces(Exception):
    pass

#mendefinisikan dua fungsi, get_landmarks dan annotate_landmarks
def get_landmarks(im): 
    # mendeteksi 1 wajah
    rects = detector(im, 1) 

    if len(rects) > 1: 
        # jika rects (hasil deteksi wajah) lebih dari 1 maka akan muncul error TooManyFaces
        raise TooManyFaces
    if len(rects) == 0: 
        #Sedangkan jika rects sama dengan 0 maka akan muncul error NoFaces.
        raise NoFaces
    
    #akan mengembalikan sebuah matrix numpy yang berisi koordinat landmark pada wajah
    return numpy.matrix([[p.x, p.y] for p in predictor(im,rects[0]).parts()]) 

#mengambil gambar dan koordinat landmark, dan mengembalikan gambar yang telah diberi tanda pada setiap landmark
def annotate_landmarks(im, landmarks): 
    im = im.copy() 
    # menambahkan tulisan dan lingkaran ke masing-masing landmark pada gambar
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        # melakukan iterasi melalui daftar landmark dan menambahkan tulisan dan lingkaran ke setiap landmark dgn putText & circle
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4, color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

image = cv2.imread('face.jpg')
landmarks = get_landmarks(image)
image_with_landmarks = annotate_landmarks(image, landmarks)
cv2.imshow('Result', image_with_landmarks)
cv2.imwrite('image_with_landmarks.jpg', image_with_landmarks)
cv2.waitKey(0)
cv2.destroyAllWindows()
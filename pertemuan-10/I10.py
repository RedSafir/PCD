import cv2
import dlib
import numpy
from time import sleep
import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS ,NOSE_POINTS + MOUTH_POINTS]

COLOUR_CORRECT_BLUR_FRAC = 0.6
cascade_path = 'haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# mendapatkan landmark pada gambar
def get_landmarks(im, dlibOn):
    if (dlibOn == True):
        rects = detector(im, 1)
        if len(rects) > 1:
            return "error"
        if len(rects) == 0:
            return "error"
        return numpy.matrix([[p.x, p.y] for p in predictor(im,rects[0]).parts()])
    else:
        rects = cascade.detectMultiScale(im, 1.3, 5)
        if len(rects) > 1:
            return "error"
        if len(rects) == 0:
            return "error"
        x, y, w, h = rects[0]
        rect = dlib.rectangle(x, y, x + w, y + h)
        return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    
# menandai landmark wajah pada gambar 
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=0.4,color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

# digunakan untuk menggambar poligon konveks dari himpunan titik
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

# digunakan untuk membuat masker wajah yang akan digunakan untuk overlay dengan wajah asli
def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
        landmarks[group], color=1)
    im = numpy.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im

# digunakan untuk menghitung transformasi geometri antara dua himpunan titik landmark
def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), numpy.matrix([0., 0., 1.])])

# dugunakan untuk membaca landmark  
def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_LINEAR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
    im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im, dlibOn)
    return im, s

#  digunakan untuk melakukan warp atau transformasi perspektif pada gambar
def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,M[:2], (dshape[1], dshape[0]),
        dst=output_im,
        borderMode=cv2.BORDER_TRANSPARENT,
        flags=cv2.WARP_INVERSE_MAP)
    return output_im

# digunakan untuk memperbaiki kesalahan warna pada wajah yang akan ditukar dengan wajah asli.
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) / im2_blur.astype(numpy.float64))

# melakukan pertukaran wajah
def face_swap(img, name):
    s = get_landmarks(img, True)
    if (s == "error"):
        print("No or too many faces")
        return img
    im1, landmarks1 = img, s
    im2, landmarks2 = read_im_and_landmarks(name)
    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1),
                               warped_mask],axis=0)
    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    cv2.imwrite('output.jpg', output_im)
    image = cv2.imread('output.jpg')
    frame = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return image

cap = cv2.VideoCapture(0)
filter_image = "jokowi.jpg"
dlibOn = False
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.75, fy=0.75,
                       interpolation=cv2.INTER_LINEAR)
    frame = cv2.flip(frame, 1)
    cv2.imshow('Our Amazing Face Swapper', face_swap(frame, filter_image))
    if cv2.waitKey(0):
        break

cap.release()
cv2.destroyAllWindows()
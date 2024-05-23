import cv2 as cv
import os
import numpy as np

img_folder = 'img_2/pts'
img_save_folder = 'img_2/pts_threshold'
folder = os.path.exists(img_save_folder)
if not folder:
    os.makedirs(img_save_folder)
img_list = os.listdir(img_folder)

def cv_imread(file_path):
    cv_img = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), cv.IMREAD_COLOR)
    return cv_img


kernel = np.ones((3,3),np.uint8)

for imgs in img_list:
    img_path = os.path.join(img_folder, imgs)
    img_save_path = os.path.join(img_save_folder, imgs)
    img = cv_imread(img_path)

    img_erode = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    ret, img_threshold = cv.threshold(img_erode, 20, 255, cv.THRESH_BINARY)
    cv.imwrite(img_save_path, img_threshold)

# cv.imshow('img.png', img)
# cv.imshow('img_blur.png', img_blur)
# cv.waitKey(0)

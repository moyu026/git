import cv2 as cv
import os
import numpy as np

img_folder = 'img/img_model/img_side'
img_save_folder = 'img/img_model/img_CLOSE'
folder = os.path.exists(img_save_folder)
if not folder:
    os.makedirs(img_save_folder)
img_list = os.listdir(img_folder)


kernel = np.ones((3,3),np.uint8)

for imgs in img_list:
    img_path = os.path.join(img_folder, imgs)
    img_save_path = os.path.join(img_save_folder, imgs)
    img = cv.imread(img_path)

    img_erode = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    cv.imwrite(img_save_path, img_erode)

# cv.imshow('img.png', img)
# cv.imshow('img_blur.png', img_blur)
# cv.waitKey(0)

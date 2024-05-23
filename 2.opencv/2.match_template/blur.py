import cv2 as cv
import os

img_folder = 'img/img_model/img_side'
img_save_folder = 'img/img_model/img_blur_7'

folder = os.path.exists(img_save_folder)
if not folder:
    os.makedirs(img_save_folder)

img_list = os.listdir(img_folder)

for imgs in img_list:
    img_path = os.path.join(img_folder, imgs)
    img_save_path = os.path.join(img_save_folder, imgs)
    img = cv.imread(img_path)

    img_blur = cv.blur(img, (7, 7))
    # img_medianBlur = cv.medianBlur(img, 5)
    cv.imwrite(img_save_path, img_blur)

# cv.imshow('img.png', img)
# cv.imshow('img_blur.png', img_blur)
# cv.waitKey(0)

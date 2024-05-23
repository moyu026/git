import cv2
import numpy as np
import os

def img_resize(img_folder, imgbg_folder):

    img_list = os.listdir(img_folder)
    for imgs in img_list:
        img_path = os.path.join(img_folder, imgs)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        imgbg_path = os.path.join(imgbg_folder, imgs)
        img_bg = cv2.imread(imgbg_path)
        resize_img_bg = cv2.cv2.resize(img_bg, (width, height))

    return img, resize_img_bg

def findContours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
    ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)  # ret:设定的阈值， binary：二值化后的图像
    # cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作
    kernel = np.ones((1, 5), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=6)
    mask_inv = cv2.bitwise_not(mask)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




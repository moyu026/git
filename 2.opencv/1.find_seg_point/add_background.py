import cv2
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

img_folder = r'F:\PythonWork\dataset\firedataset\1\rgb_black'
save_folder = r'F:\PythonWork\dataset\firedataset\1\img_seg'
img_list = os.listdir(img_folder)

img_path = r'F:\PythonWork\4.opencv\1.find_seg_point\img\013.png'
background_img_path = r'F:\PythonWork\4.opencv\1.find_seg_point\background\1.jpg'
img = cv2.imread(img_path)
background_img = cv2.imread(background_img_path)

height, width = img.shape[:2]

resize_background_img = cv2.resize(background_img, (width, height))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片

ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)  # ret:设定的阈值， binary：二值化后的图像

# cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作
kernel = np.ones((1, 5), np.uint8)
mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=6)
cv2.imshow("mask", mask)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow("mask_inv", mask_inv)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

n = []
max_contours = []
for i in contours:
    area = cv2.contourArea(i)
    if area > 10000:
        n.append(area)
        max_contours.append(i)

        # with open('point.txt', 'w') as f:
        #     f.write(str(i))

n.sort(reverse=True)

pts = max_contours

# 逐位与，得到裁剪后图像，此时是黑色背景
dst = cv2.bitwise_and(img, img, mask=mask)

bkg = cv2.bitwise_and(resize_background_img, resize_background_img, mask=mask_inv)

final = dst + bkg

# cv2.drawContours(img, max_contours, -facilities_img, (0, 0, 255), 3)
# cv2.imshow("mask.jpg", mask)
cv2.imshow("dst.jpg", dst)
cv2.imshow("bkg.jpg", bkg)
cv2.imshow('bgimg', final)

cv2.waitKey(0)

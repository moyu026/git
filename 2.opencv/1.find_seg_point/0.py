import random

import cv2
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

img_folder = r'F:\PythonWork\dataset\firedataset\fire_bg\img_black'
save_folder = r'F:\PythonWork\dataset\firedataset\fire_bg\img_white'
txt_folder = r'F:\PythonWork\dataset\firedataset\fire_bg\txt_point'
img_list = os.listdir(img_folder)


img_path = r'img/002.png'
img_bg_path = r'background/1.jpg'

txt_path = 'point.txt'

img = cv2.imread(img_path)
img_bg = cv2.imread(img_bg_path)

rows, cols, channels = img.shape
rows_bg, cols_bg, channels_bg = img_bg.shape

rows_dif, cols_dif = rows_bg - rows, cols_bg - cols
# row_random, cols_random = random.randint(0, rows_dif), random.randint(0, cols_dif)
row_random, cols_random = 15, 15

# print(row_random, cols_random)

roi = img_bg[row_random:rows+row_random, cols_random:cols+cols_random]


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)  # 二值化， ret:设定的阈值， binary：二值化后的图像
kernel = np.ones((1, 5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=6) # cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 检测轮廓，contours：轮廓坐标 ，hierarchy：每条轮廓对应的属性

max_contours = []
for i in contours:
    area = cv2.contourArea(i)
    if area > 10000:
        max_contours.append(i)
pts = max_contours
pts1 = pts

# 一块区域
# x = []
# y = []
# # print(len(pts[0]))
# for idx in range(len(pts1[0])):
#     # print(pts1[0][idx][0][0])
#     pts1[0][idx][0][0] = pts1[0][idx][0][0] + row_random
#     pts1[0][idx][0][facilities_img] = pts1[0][idx][0][facilities_img] + cols_random
# # print(pts1)
#     x.append(pts1[0][idx][0][0]/rows_bg)
#     y.append(pts1[0][idx][0][facilities_img]/cols_bg)
# point = zip(x, y)
#
# with open('point1.txt', 'w') as f:
#     for a in point:
#         f.write(str(a))






# with open(txt_path, 'w') as f:
#     f.write(str(pts))
#     print(txt_path)


# 创建区域为pts的mask
mask = np.zeros(img.shape[:2], np.uint8)
cv2.polylines(mask, pts, 1, 255)  # 描绘边缘  在mask上将多边形区域填充为白色
cv2.fillPoly(mask, pts, 255)  # 填充


mask_inv = cv2.bitwise_not(mask)
fire = cv2.bitwise_and(img, img, mask=mask)
# 带mask的部分背景
bkg_roi = cv2.bitwise_and(roi, roi, mask=mask_inv)
# 前景和背景相加
dst = cv2.add(fire, bkg_roi)
# 添加到完整的背景图上
img_bg[row_random:rows+row_random, cols_random:cols+cols_random] = dst


cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('img', img_bg)
cv2.imshow('fire', fire)
cv2.imwrite('mask.jpg', mask)

cv2.waitKey(0)






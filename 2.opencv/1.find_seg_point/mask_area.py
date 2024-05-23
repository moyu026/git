import cv2
import numpy as np
import os

img_folder = r'F:\PythonWork\dataset\firedataset\1\rgb_black'
save_folder = r'F:\PythonWork\dataset\firedataset\1\img_seg'
img_list = os.listdir(img_folder)

kernel = np.ones((1, 5), np.uint8)

img_path = 'img/black1.png'
background_img_path = 'background/1.jpg'
img = cv2.imread(img_path)
background_img = cv2.imread(background_img_path)

height, width = img.shape[:2]
print(background_img.shape)

resize_background_img = cv2.resize(background_img, (width, height))

# cv2.imwrite('bkg.jpg', resize_background_img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片

# 将图像二值化
# 第一个参数是源图像，应该是灰度图；
# 第二个参数是对图像进行分类的阈值；
# 第三个参数是最大值，表示如果像素值大于（有时小于）阈值则要给出的值
ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY) # ret:设定的阈值， binary：二值化后的图像

# cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=5)


contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

n = []
max_contours = []
for i in contours:
    area = cv2.contourArea(i)
    if area > 10000:
        max_contours.append(i)

pts = max_contours
# 和原始图像一样大小的0矩阵，作为mask
mask = np.zeros(img.shape[:2], np.uint8)



# 在mask上将多边形区域填充为白色
cv2.polylines(mask, pts, 1, 255)  # 描绘边缘
cv2.fillPoly(mask, pts, 255)  # 填充
# 逐位与，得到裁剪后图像，此时是黑色背景
dst = cv2.bitwise_and(img, img, mask=mask)
# 添加白色背景
bg = np.ones_like(img, np.uint8) * 255
cv2.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
dst_white = bg + dst

# bkgimg = cv2.add(dst, resize_background_img)



# cv2.drawContours(img, max_contours, -facilities_img, (0, 0, 255), 3)
# cv2.imshow("mask.jpg", mask)
# cv2.imshow("dst.jpg", dst)
cv2.imshow("dst_white.jpg", dst_white)

# cv2.imshow('bgimg',bkgimg)

cv2.waitKey(0)
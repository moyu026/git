import cv2
import numpy as np
import os

img_path = r'F:\PythonWork\3.opencv\4.electircal cabinet\img\11\1.jpg'

kernel = np.ones((5, 5), np.uint8)

img = cv2.imread(img_path)
# image = cv2.blur(img, (3, 3))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片

# 将图像二值化
# 第一个参数是源图像，应该是灰度图；
# 第二个参数是对图像进行分类的阈值；
# 第三个参数是最大值，表示如果像素值大于（有时小于）阈值则要给出的值
ret, binary = cv2.threshold(gray, 90, 150, cv2.THRESH_BINARY) # ret:设定的阈值， binary：二值化后的图像

# cv2.imshow("img", binary)
# cv2.waitKey(0)

# cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=5)

contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
# cv2.imwrite('contours.jpg', img)

cv2.imshow("img", img)
cv2.waitKey(0)
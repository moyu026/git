# coding:utf8

import cv2
import numpy as np


def row_method(src):
    image = np.array(src)
    cimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度图
    circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 40, param1=250, param2=58, minRadius=0)
    circles = np.uint16(np.around(circles))  # 取整
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 在原图上画圆，圆心，半径，颜色，线框
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)  # 画圆心
    cv2.putText(image, "param1=250, param2=58", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imwrite("row_circles.jpg", image)


def threshold_OTSU_method(src):
    image = np.array(src)
    cimage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 灰度图
    th, dst = cv2.threshold(cimage, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_TRUNC + cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=47, minRadius=0)
    circles = np.uint16(np.around(circles))  # 取整
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 在原图上画圆，圆心，半径，颜色，线框
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)  # 画圆心
    cv2.putText(image, "param1=50, param2=47", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imwrite("otsu_circles.jpg", image)


def threshold_triangle_method(src):
    image = np.array(src)
    cimage = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 灰度图
    th, dst = cv2.threshold(cimage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=17, minRadius=0)
    circles = np.uint16(np.around(circles))  # 取整
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 在原图上画圆，圆心，半径，颜色，线框
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)  # 画圆心
    cv2.putText(image, "param1=50, param2=17", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imwrite("triangle_circles.jpg", image)


def mean_circles(src):
    image = np.array(src)
    dst = cv2.pyrMeanShiftFiltering(image, 10, 100)  # 均值偏移滤波
    cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)  # 灰度图
    circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=20, minRadius=0)
    circles = np.uint16(np.around(circles))  # 取整
    for i in circles[0, :]:
        cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 2)  # 在原图上画圆，圆心，半径，颜色，线框
        cv2.circle(image, (i[0], i[1]), 2, (255, 0, 0), 2)  # 画圆心

    cv2.putText(image, "param1=50, param2=20", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imwrite("mean_circles.jpg", image)


src = cv2.imread("img_test/5.jpg")  # 读取图片位置
# cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("input image", src)

threshold_OTSU_method(src)
threshold_triangle_method(src)
mean_circles(src)
row_method(src)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2 as cv
import numpy as np

img1 = cv.imread("gray.png")

img_point = cv.imread("img/test/2.1.png")


gray_img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
gray_img2 = cv.cvtColor(img_point, cv.COLOR_RGB2GRAY)



cv.imshow('gray_img1', gray_img1)
cv.imshow('gray_img2', gray_img2)
cv.imwrite('gray1.png', gray_img1)
# cv.imwrite('gray_2.1.png', gray_img2)

# cv.waitKey(0)
import cv2 as cv
import numpy as np



img_path = r'img_2/obj_threshold/1-3obj.png'

img = cv.imread(img_path)
img = img[185:910, 420:960]

# cv.imshow("img", img)
# cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, binary = cv.threshold(gray, 254, 255, cv.THRESH_BINARY)  # ret:设定的阈值， binary：二值化后的图像

kernel = np.ones((1, 5), np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)

contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


x = []
y = []
# print(len(contours[0]))
for i in range(len(contours[0])):
    x.append(contours[0][i][0][0])
    y.append(contours[0][i][0][1])

min_x, max_x = min(x), max(x)
min_y, max_y = min(y), max(y)

img1 = img[min_y-10:max_y+10, min_x-10:max_x+10]

print(min(x), max(x))
print(y)


# min_val, max_val, min_loc, max_loc = cv.minMaxLoc(contours)

# print(min_val)
# print(max_val)

cv.drawContours(img1, contours, -1, (0, 0, 255), 3)

cv.imshow("img", img1)
cv.waitKey(0)
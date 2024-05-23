import cv2
import numpy as np
img_path = r'F:\PythonWork\3.opencv\4.electircal cabinet\img_test\button\button2-3.jpg'

image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, img_threshold = cv2.threshold(gray, 40, 60, cv2.THRESH_BINARY)
cv2.imwrite('edge.jpg', )
kernel = np.ones((5, 5), np.uint8)

# 进行边缘检测
edges = cv2.Canny(gray, 50, 200)

edges = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.erode(edges, kernel, iterations=2)
cv2.imwrite('edge.jpg', edges)


# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_contours = []
for i in contours:
    area = cv2.contourArea(i)
    if area > 5000:
        max_contours.append(i)

# # print(max_contours[0])
# x = []
# y = []
# for points in max_contours[0]:
#     x.append(points[0][0])
#     y.append(points[0][1])
#     if
#
#     # print(points[0])
# print(max(x))
# print(y)

cv2.drawContours(image, max_contours, -1, (0, 0, 255), 3)
cv2.imwrite('save_img2.jpg', image)

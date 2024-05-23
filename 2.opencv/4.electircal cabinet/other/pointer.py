import cv2
import numpy as np

img_path = 'img/pointer1.jpg'
image = cv2.imread(img_path)


# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('gray.jpg', gray)
_, gray = cv2.threshold(gray, 150, 200, cv2.THRESH_BINARY)
cv2.imwrite('threshold1.jpg', gray)

# # 边缘检测
# edges = cv2.Canny(gray, 130, 140, apertureSize=3)
# cv2.imwrite('edge.jpg', edges)
#
#
#
# lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold=10, minLineLength=1, maxLineGap=1)
#
# for x1, y1, x2, y2 in lines[0]:
#     print('x1:', x1, 'y1:', y1, 'x2:', x2, 'y2:', y2)
#     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# cv2.imwrite('line.jpg', image)

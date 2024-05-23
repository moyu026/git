import cv2
import numpy as np
# 读取图像
image = cv2.imread('img_test/test2.jpg')

# 将图像转换为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 边缘检测
edges = cv2.Canny(gray, 180, 200)

kernel = np.ones((3, 3), np.uint8)
# dilated = cv2.dilate(edges, kernel, iterations=1)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=3)  #CLOSE:先膨胀，再腐蚀

# cv2.imwrite('output/test/dilate1.jpg', edges)

# 轮廓检测
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
# cv2.imwrite('contourtest.jpg', image)

# 遍历轮廓
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 200:
        # 多边形逼近
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 拟合旋转矩形
        rect = cv2.minAreaRect(contour)

        # 筛选条件
        area = rect[1][0] * rect[1][1]
        # aspect_ratio = max(rect[1]) / min(rect[1])

        # if area > 1000 and aspect_ratio < 2:
            # 绘制矩形
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

# cv2.imwrite('output/test/line2.jpg', image)
# 显示结果
cv2.imshow('Rectangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



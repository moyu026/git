import cv2
import numpy as np

# 读取图像
image = cv2.imread('img_test/1.jpg')  # 替换为你的图像文件路径
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, 150, 200)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓并检测三角形
for contour in contours:
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 3:
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

# 显示结果图像
cv2.imshow('Triangles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

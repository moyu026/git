import cv2
import numpy as np


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# 读取图像
image = cv2.imread('contour_image.png', cv2.IMREAD_GRAYSCALE)

# 寻找轮廓
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 计算每个轮廓的中心点
contour_centers = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contour_centers.append((cX, cY))

# 遍历每对轮廓并判断距离条件
for i in range(len(contour_centers)):
    for j in range(i + 1, len(contour_centers)):
        distance = calculate_distance(contour_centers[i], contour_centers[j])
        vertical_distance = abs(contour_centers[i][1] - contour_centers[j][1])
        horizontal_distance = abs(contour_centers[i][0] - contour_centers[j][0])

        if vertical_distance < 10 and horizontal_distance < 20:
            print(f"Contours {i} and {j} meet the distance criteria.")

            # 根据轮廓的位置绘制不同颜色
            if contour_centers[i][0] < contour_centers[j][0]:
                cv2.drawContours(image, [contours[i]], -1, (0, 0, 255), 2)  # 红色
                cv2.drawContours(image, [contours[j]], -1, (255, 0, 0), 2)  # 蓝色
            else:
                cv2.drawContours(image, [contours[i]], -1, (255, 0, 0), 2)  # 蓝色
                cv2.drawContours(image, [contours[j]], -1, (0, 0, 255), 2)  # 红色

cv2.imshow('Contour Centers', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# 读取图像
image = cv2.imread('img/5.jpg')

# 将图像从BGR转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置绿色颜色的HSV范围
lower_green = np.array([40, 80, 80])
upper_green = np.array([150, 255, 255])

# 创建一个遮罩，只保留在HSV范围内的绿色区域
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 对遮罩进行形态学操作，消除噪声并填充区域
kernel = np.ones((5, 5), np.uint8)
# green_mask = cv2.erode(green_mask, kernel, iterations=1)
# green_mask = cv2.dilate(green_mask, kernel, iterations=1)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=5)

# 寻找绿色区域的轮廓
contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 遍历所有轮廓
for contour in contours:
    # 计算轮廓的中心坐标
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])

        # 输出绿色区域的中心坐标
        print("绿色区域的中心坐标: ({}, {})".format(center_x, center_y))

# 在原始图像上通过遮罩提取绿色区域
green_region = cv2.bitwise_and(image, image, mask=green_mask)

# 显示提取的绿色区域
cv2.imwrite('GreenRegion.jpg', green_region)
cv2.waitKey(0)
cv2.destroyAllWindows()

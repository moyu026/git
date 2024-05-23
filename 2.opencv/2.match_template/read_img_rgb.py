import cv2
import numpy as np

# 读取图像
image = cv2.imread('eletrical cabinet/4.jpg')

# 将图像从BGR转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置绿色颜色的HSV范围rgb(202,255,13)
lower_green = np.array([30, 40, 30])
upper_green = np.array([25, 250, 25])

# 创建一个遮罩，只保留在HSV范围内的绿色区域
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 对遮罩进行形态学操作，消除噪声并填充区域
kernel = np.ones((5, 5), np.uint8)
green_mask = cv2.erode(green_mask, kernel, iterations=1)
green_mask = cv2.dilate(green_mask, kernel, iterations=1)

# 在原始图像上通过遮罩提取绿色区域
green_region = cv2.bitwise_and(image, image, mask=green_mask)

# 显示提取的绿色区域
# cv2.imshow('Green Region', green_region)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('locate.jpg', green_region)

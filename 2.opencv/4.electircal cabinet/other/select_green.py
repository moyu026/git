import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'..\img/small_light/2.jpg')

# 将图像从BGR转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置绿色颜色的HSV范围
lower_green = np.array([40, 80, 80])
upper_green = np.array([150, 255, 255])

# 创建一个遮罩，只保留在HSV范围内的绿色区域
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 对遮罩进行形态学操作，消除噪声并填充区域
kernel = np.ones((5, 5), np.uint8)
green_mask = cv2.erode(green_mask, kernel, iterations=1)
green_mask = cv2.dilate(green_mask, kernel, iterations=1)

# 在原始图像上通过遮罩提取绿色区域
green_region = cv2.bitwise_and(image, image, mask=green_mask)

# 显示提取的绿色区域
cv2.imwrite('GreenRegion.jpg', green_region)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
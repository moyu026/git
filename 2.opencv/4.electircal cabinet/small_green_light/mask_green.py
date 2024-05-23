import cv2
import numpy as np
import os

img_folder = r'F:\PythonWork\dataset\electrical_cabinet_all\detect\5.small_lights\add_small_light\extea_little_light'
img_list =os.listdir(img_folder)

image = cv2.imread(r'..\img/small_light/2.jpg')

# 将图像从BGR转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置绿色颜色的HSV范围
# lower_green = np.array([40, 80, 80])
# upper_green = np.array([150, 255, 255])
lower_green = np.array([40, 80, 80])
upper_green = np.array([100, 255, 255])

green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

kernel = np.ones((5, 5), np.uint8)
green_mask = cv2.erode(green_mask, kernel, iterations=1)
green_mask = cv2.dilate(green_mask, kernel, iterations=1)

green_region = cv2.bitwise_and(image, image, mask=green_mask)

cv2.imwrite('GreenRegion.jpg', green_region)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
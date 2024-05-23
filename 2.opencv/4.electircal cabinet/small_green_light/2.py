import cv2
import numpy as np

'''
计算亮灯区域的HSV均值，判断是否在亮灯的HSV范围内
计算不亮灯区域的HSV均值，判断是否在不亮灯的HSV范围内
'''
# 读取图像
image = cv2.imread(r'..\img/small_light/2.jpg')
image = image[402:432, 804:835]

# 转换图像为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义HSV范围
lower_green = np.array([40, 80, 80])
upper_green = np.array([150, 255, 255])

mask = cv2.inRange(hsv_image, lower_green, upper_green)

# cv2.imshow('i0', image)
# cv2.imshow('img', mask)
# cv2.waitKey()

hsv_mean = cv2.mean(hsv_image, mask=mask)

print(f'Mean Hue (色相): {hsv_mean[0]}')
print(f'Mean Saturation (饱和度): {hsv_mean[1]}')
print(f'Mean Value (明度): {hsv_mean[2]}')

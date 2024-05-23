import cv2
import numpy as np

# 定义一个回调函数来处理鼠标事件
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取鼠标位置的HSV值
        hsv_value = hsv_image[y, x]

        # 打印HSV值
        print(f'Hue (色相): {hsv_value[0]}')
        print(f'Saturation (饱和度): {hsv_value[1]}')
        print(f'Value (明度): {hsv_value[2]}')

image = cv2.imread(r'..\img/small_light/2.jpg')
image = cv2.resize(image, dsize=(1024, 1024), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

# 转换图像为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 创建一个窗口并设置鼠标回调
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', image)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下Esc键退出循环
        break

cv2.destroyAllWindows()
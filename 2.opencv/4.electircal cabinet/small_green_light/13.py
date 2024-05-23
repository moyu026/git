import cv2
import numpy as np

# 定义一个回调函数来处理鼠标事件
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        is_in_range = cv2.inRange(hsv_image, lower_green, upper_green)
        '''
        cv2.inRange() 函数返回一个二值掩码图像，其中像素的值要么为0，要么为255。
        如果像素的HSV值在指定的范围内，则该像素在二值掩码图像中的值将被设置为255。
        如果像素的HSV值不在指定的范围内，则该像素在二值掩码图像中的值将被设置为0。
        '''

        # 判断指定位置的像素是否在范围内
        if is_in_range[y, x] == 255:
            print("HSV值在指定范围内。")
        else:
            print("HSV值不在指定范围内。")


image = cv2.imread(r'..\img/small_light/4.jpg')
image = cv2.resize(image, dsize=(1024, 1024), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

# 转换图像为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义HSV范围
lower_green = np.array([40, 80, 80])
upper_green = np.array([150, 255, 255])

# 创建一个窗口并设置鼠标回调
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

while True:
    cv2.imshow('Image', image)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下Esc键退出循环
        break

cv2.destroyAllWindows()
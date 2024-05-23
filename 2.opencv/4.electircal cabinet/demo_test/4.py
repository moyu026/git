import cv2
import numpy as np
from paddleocr import PaddleOCR

# ocr
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 读取图像
image = cv2.imread(r'img/demo_test/light1.jpg')

# 将图像从BGR转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置绿色颜色的HSV范围
lower_green = np.array([20, 100, 100])
upper_green = np.array([50, 255, 255])

# 创建一个遮罩，只保留在HSV范围内的绿色区域
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
# cv2.imshow('g', green_mask)
# cv2.waitKey()

kernel = np.ones((5, 5), np.uint8)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=1)

# 寻找绿色区域的轮廓
contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个和原始图像大小相同的图像，用于绘制矩形
rectangle_image = np.copy(image)

with open('ocr.txt', 'a') as f:
    for contour in contours:
        # 每个轮廓上的最右点坐标
        right_point = tuple(contour[contour[:, :, 0].argmax()][0])

        rectangle_left = right_point[0] + 20
        rectangle_top = right_point[1] - 20

        txt_img = rectangle_image[rectangle_top:rectangle_top + 50, rectangle_left:rectangle_left + 230]

        # ocr识别
        result_ocr = ocr.ocr(txt_img, cls=True)
        print(result_ocr)
        f.write(str(result_ocr[0][0][1][0]) + '\n')

    # 绘制红色矩形
        cv2.rectangle(rectangle_image, (rectangle_left, rectangle_top), (rectangle_left + 230, rectangle_top + 50), (0, 0, 255), thickness=2)

cv2.imwrite('re.jpg', rectangle_image)












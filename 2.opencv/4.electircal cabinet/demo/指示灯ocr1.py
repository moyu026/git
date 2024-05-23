import cv2
import numpy as np
from paddleocr import PaddleOCR

# ocr
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 读取图像
image = cv2.imread(r'F:\PythonWork\3.opencv\4.electircal cabinet\img\demo_test\light.jpg')

# 将图像从BGR转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置绿色颜色的HSV范围
lower_green = np.array([20, 200, 200])
upper_green = np.array([50, 255, 255])

# 创建一个遮罩，只保留在HSV范围内的绿色区域
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

kernel = np.ones((5, 5), np.uint8)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=1)

# 寻找绿色区域的轮廓
contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个和原始图像大小相同的图像，用于绘制矩形
rectangle_image = np.copy(image)

with open('ocrtxt.txt', 'a') as f:
    for contour in contours:
        # 计算每个轮廓的中心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        # 左边的txt
        left_rectangle_right = cX - 20
        left_rectangle_bottom = cY + 20
        left_txt1 = rectangle_image[left_rectangle_bottom - 50:left_rectangle_bottom, left_rectangle_right - 70:left_rectangle_right]

        # ocr
        left_ocr1 = ocr.ocr(left_txt1, cls=True)

        if len(left_ocr1[0]) != 0:
            left_txt2 = rectangle_image[left_rectangle_bottom - 50:left_rectangle_bottom, left_rectangle_right - 130:left_rectangle_right]
            left_ocr2 = ocr.ocr(left_txt2, cls=True)
            print(left_ocr2)

            cv2.rectangle(rectangle_image, (left_rectangle_right - 130, left_rectangle_bottom - 50), (left_rectangle_right, left_rectangle_bottom),(0, 0, 255), thickness=2)
            # f.write(str(left_ocr2[0][0][1][0]) + '\n')

        # 右边的txt
        right_rectangle_left = cX + 20
        right_rectangle_top = cY - 20
        right_txt1 = rectangle_image[right_rectangle_top:right_rectangle_top + 50, right_rectangle_left:right_rectangle_left + 70]
        # ocr
        right_txt1 = ocr.ocr(right_txt1, cls=True)

        if len(right_txt1[0]) != 0:
            right_txt2 = rectangle_image[right_rectangle_top:right_rectangle_top + 50, right_rectangle_left:right_rectangle_left + 130]
            right_ocr2 = ocr.ocr(right_txt2)
            print(right_ocr2)

            cv2.rectangle(rectangle_image, (right_rectangle_left, right_rectangle_top), (right_rectangle_left + 130, right_rectangle_top + 50), (0, 0, 255), thickness=2)

# cv2.imwrite('ocrtxt.jpg', rectangle_image)


cv2.imshow('Contour Centers', rectangle_image)
cv2.waitKey(0)
cv2.destroyAllWindows()











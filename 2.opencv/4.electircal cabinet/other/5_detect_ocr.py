import cv2
import numpy as np
import os
from paddleocr import PaddleOCR, draw_ocr

# ocr
ocr = PaddleOCR(use_angle_cls=True, lang="ch")


# 读取图像
image = cv2.imread('img/demo_test/light1.jpg')
save_folder = 'output'

# 将图像从BGR转换为HSV颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 设置绿色颜色的HSV范围
lower_green = np.array([70, 80, 80])
upper_green = np.array([150, 255, 255])

# 创建一个遮罩，只保留在HSV范围内的绿色区域
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
cv2.imshow('g', green_mask)
cv2.waitKey()

# # 对遮罩进行形态学操作，消除噪声并填充区域
# kernel = np.ones((5, 5), np.uint8)
# green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=3)
#
# # 寻找绿色区域的轮廓
# contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # 创建一个和原始图像大小相同的图像，用于绘制矩形
# rectangle_image = np.copy(image)
#
# count_txt = 0
#
# with open('ocr.txt', 'a')as f:
#     for contour in contours:
#
#         # 每个轮廓上的顶点坐标
#         top_point = tuple(contour[contour[:, :, 1].argmin()][0])
#         # print("绿色区域的最上部坐标: ({}, {})".format(top_point[0], top_point[1]))
#
#         rectangle_right = top_point[0] + 140
#         rectangle_bottom = top_point[1] - 10
#
#         # 绘制红色矩形
#     #     cv2.rectangle(rectangle_image, (right, bottom), (right - 230, bottom - 80), (0, 0, 255), thickness=2)
#     #
#     # cv2.imwrite('re.jpg', rectangle_image)
#
#         txt_img = rectangle_image[rectangle_bottom-80:rectangle_bottom, rectangle_right-230:rectangle_right]
#
#         txt_img_path = os.path.join(save_folder, str(count_txt) + '.jpg')
#
#         count_txt = count_txt + 1
#
#         # ocr识别
#         result_ocr = ocr.ocr(txt_img, cls=True)
#         # f.write(str(result_ocr[0][0][1][0]) + '\n')  #输出的名称
#         f.write(str(result_ocr[0][0][0]) + '\n')  # 输出字符的坐标
        # print(result_ocr[0][0])



# # 在原始图像上通过遮罩提取绿色区域
# green_region = cv2.bitwise_and(image, image, mask=green_mask)
#
# cv2.imwrite('mask.jpg', green_region)

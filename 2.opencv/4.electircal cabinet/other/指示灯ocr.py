import cv2
import numpy as np
# from paddleocr import PaddleOCR

# ocr
# ocr = PaddleOCR(use_angle_cls=True, lang="ch")

# 读取图像
image = cv2.imread('img_test/ocr1-2-2.jpg')

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

# 计算每个轮廓的中心点
contour_centers = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        contour_centers.append((cX, cY))

# 初始化最小竖直距离和对应的轮廓索引
min_vertical_distance = float('inf')
closest_contours = None

rectangle_image = np.copy(image)

# 遍历每对轮廓并计算竖直距离
for i in range(len(contour_centers)):
    for j in range(i + 1, len(contour_centers)):
        vertical_distance = abs(contour_centers[i][1] - contour_centers[j][1])
        horizontal_distance = abs(contour_centers[i][0] - contour_centers[j][0])

        if vertical_distance < 40 and horizontal_distance < 130:
            print(f"Contours {i} and {j} meet the distance criteria.")

            # 判断左右位置
            if contour_centers[i][0] < contour_centers[j][0]:
                left_rectangle_left = contour_centers[i][0] - 140
                left_rectangle_top = contour_centers[i][1] - 20
                left_txt = rectangle_image[left_rectangle_top:left_rectangle_top + 50, left_rectangle_left:left_rectangle_left + 130]
                cv2.rectangle(rectangle_image, (left_rectangle_left, left_rectangle_top), (left_rectangle_left + 130, left_rectangle_top + 50), (0, 0, 255), thickness=2)

                rigth_rectangle_left = contour_centers[j][0] + 30
                rigth_rectangle_top = contour_centers[j][1] - 30
                right_txt = rectangle_image[rigth_rectangle_top:rigth_rectangle_top + 55, rigth_rectangle_left:rigth_rectangle_left + 120]
                cv2.rectangle(rectangle_image, (rigth_rectangle_left, rigth_rectangle_top), (rigth_rectangle_left + 130, rigth_rectangle_top + 50), (0, 0, 255), thickness=2)

            else:
                left_rectangle_left = contour_centers[j][0] - 140
                left_rectangle_top = contour_centers[j][1] - 20
                left_txt = rectangle_image[left_rectangle_top:left_rectangle_top + 50, left_rectangle_left:left_rectangle_left + 130]
                cv2.rectangle(rectangle_image, (left_rectangle_left, left_rectangle_top), (left_rectangle_left + 130, left_rectangle_top + 50), (0, 0, 255), thickness=2)

                rigth_rectangle_left = contour_centers[i][0] + 30
                rigth_rectangle_top = contour_centers[i][1] - 30
                right_txt = rectangle_image[rigth_rectangle_top:rigth_rectangle_top + 55, rigth_rectangle_left:rigth_rectangle_left + 120]
                cv2.rectangle(rectangle_image, (rigth_rectangle_left, rigth_rectangle_top), (rigth_rectangle_left + 130, rigth_rectangle_top + 50), (0, 0, 255), thickness=2)

        else:
            rectangle_left = contour_centers[i][0] - 70
            rectangle_top = contour_centers[i][1] - 20
            txt_img = rectangle_image[rectangle_top:rectangle_top + 50, rectangle_left:rectangle_left + 130]
            cv2.rectangle(rectangle_image, (rectangle_left, rectangle_top), (rectangle_left + 70, rectangle_top + 50), (0, 0, 255), thickness=2)






cv2.imshow('Contour Centers', rectangle_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 创建一个和原始图像大小相同的图像，用于绘制矩形
# rectangle_image = np.copy(image)
#
# for contour in contours:
#
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         # cv2.circle(rectangle_image, (cX, cY), 5, (255, 255, 255), -1)
#
#         rectangle_left = cX - 140
#         rectangle_top = cY - 20
#
#         txt_img = rectangle_image[rectangle_top:rectangle_top + 50, rectangle_left:rectangle_left + 130]
#
#         # ocr识别
#         # result_ocr = ocr.ocr(txt_img, cls=True)
#         # print(result_ocr)
#
#     # f.write(str(result_ocr[0][0][1][0]) + '\n')
#
#         # 绘制红色矩形
#         cv2.rectangle(rectangle_image, (rectangle_left, rectangle_top), (rectangle_left + 130, rectangle_top + 50), (0, 0, 255), thickness=2)
#
# cv2.imwrite('rec.jpg', rectangle_image)










# # 在原始图像上通过遮罩提取绿色区域
# green_region = cv2.bitwise_and(image, image, mask=green_mask)
#
# # 显示提取的绿色区域
# cv2.imwrite('GreenRegion.jpg', green_region)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

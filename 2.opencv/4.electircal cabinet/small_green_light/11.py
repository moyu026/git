import cv2
import numpy as np

# Load your image
image = cv2.imread(r'..\img/small_light/2.jpg')
image = cv2.GaussianBlur(image, (3, 3), sigmaX=1)
image_with_contours = image.copy()
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_green = np.array([40, 80, 80])
upper_green = np.array([150, 255, 255])
# lower_green = np.array([40, 10, 10])
# upper_green = np.array([100, 255, 80])

mask = cv2.inRange(hsv_image, lower_green, upper_green)

contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours[1]:
    # 计算轮廓的矩
    M = cv2.moments(contour)

    # 计算中心点坐标
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # 在图像上绘制中心点
    cv2.circle(image_with_contours, (cX, cY), 5, (0, 0, 255), -1)  # 画一个红色的点表示中心

cv2.drawContours(image_with_contours, contours[1], -1, (0, 0, 255), 2)

cv2.imwrite('3.jpg', image_with_contours)
# cv2.imshow("Image with Green Contours", image_with_contours)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

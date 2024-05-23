import os
import cv2
import numpy as np

img_folder = r'F:\PythonWork\dataset\electrical_cabinet_all\detect\5.small_lights\hsv_result\hsv_test'
save_img_folder = r'F:\PythonWork\dataset\electrical_cabinet_all\detect\5.small_lights\hsv_result\green_light\11'
img_list = os.listdir(img_folder)


for imgs in img_list:
    img_path = os.path.join(img_folder, imgs)
    save_img_path = os.path.join(save_img_folder, imgs)
    img = cv2.imread(img_path)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 80, 80])
    upper_green = np.array([150, 255, 255])

    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contours = img.copy()
    cv2.drawContours(image_with_contours, contours[1], -1, (0, 0, 255), 2)

    cv2.imwrite(save_img_path, image_with_contours)
    print(save_img_path)
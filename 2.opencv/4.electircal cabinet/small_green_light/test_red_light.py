import cv2
import numpy as np
import os

img_folder = r'F:\PythonWork\dataset\electrical_cabinet_all\detect\5.small_lights\test_small_light_img'
save_img_folder = r'F:\PythonWork\dataset\electrical_cabinet_all\detect\5.small_lights\test_result'

img_list = os.listdir(img_folder)
for imgs in img_list:
    img_path = os.path.join(img_folder, imgs)
    save_img_path = os.path.join(save_img_folder, imgs)


    # Load your image
    image = cv2.imread(img_path)  # Replace with the path to your image

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red color in HSV
    # lower_red = np.array([0, 100, 100])
    # upper_red = np.array([20, 255, 255])
    lower_red = np.array([40, 80, 80])
    upper_red = np.array([150, 255, 255])

    # Create a mask for the red region
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Find contours in the mask
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image in red
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours[1], -1, (0, 0, 255), 2)  # Red color

    cv2.imwrite(save_img_path, image_with_contours)
    print(save_img_path)

# # Display the image with red contours
# cv2.imshow("Image with red Contours", image_with_contours)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

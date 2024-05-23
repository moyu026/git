import cv2
import numpy as np

# Load your image
image = cv2.imread(r'..\img/small_light/2.jpg')  # Replace with the path to your image

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
'''
H:红色为0°，绿色为120°,蓝色为240,它们的补色是：黄色为60°，青色为180°,品红为300°
S:饱和度S表示颜色接近光谱色的程度。一种颜色，可以看成是某种光谱色与白色混合的结果。其中光谱色所占的比例愈大，颜色接近光谱色的程度就愈高，颜色的饱和度也就愈高。
  饱和度高，颜色则深而艳。光谱色的白光成分为0，饱和度达到最高。通常取值范围为0%～100%，值越大，颜色越饱和。
V:明度表示颜色明亮的程度，对于光源色，明度值与发光体的光亮度有关；对于物体色，此值和物体的透射比或反射比有关。通常取值范围为0%（黑）到100%（白）。
'''

# Define the lower and upper bounds for red color in HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([20, 255, 255])

# Create a mask for the red region
mask = cv2.inRange(hsv_image, lower_red, upper_red)

# Find contours in the mask
contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image in red
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours[1], -1, (0, 0, 255), 2)  # Red color

# Display the image with red contours
cv2.imshow("Image with red Contours", image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()
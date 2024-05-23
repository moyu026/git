import cv2
import numpy as np

img_path = r'F:\PythonWork\4.opencv\1.find_seg_point\facilities\depth_img\val\160-dpt_beit_large_512.png'

save_folder = r'F:\PythonWork\4.opencv\1.find_seg_point\facilities\mask\val'

save_path = save_folder + '\\' + img_path.split('\\')[-1]

# 加载图像
image = cv2.imread(img_path, 0)  # 这里假设你的图像文件名为image.jpg

# 对图像进行二值化
_, thresholded_image = cv2.threshold(image, 95, 255, cv2.THRESH_BINARY)


cv2.imwrite(save_path, thresholded_image)
# 显示二值化图像
# cv2.imshow('Threshold', thresholded_image)
# cv2.waitKey(0)



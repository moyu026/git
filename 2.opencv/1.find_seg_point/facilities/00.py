import os
import cv2
import numpy as np

img_path = r'/facilities/facilities_img\10.JPG'
mask_path = r'F:\PythonWork\4.opencv\1.find_seg_point\facilities\mask\mask.jpg'

img1 = cv2.imread(img_path)
img2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


dst = cv2.bitwise_and(img1, img1, mask=img2)


cv2.imwrite('im2.jpg', dst)
# cv2.imshow('im2', dst)
# cv2.waitKey(0)

# h,w,c = img1.shape
# img3 = np.zeros((h,w,4))
# img3[:,:,0:3] = img1
# img3[:,:,3] = img2
# #这里命名随意，但是要注意使用png格式
# cv2.imwrite(r'F:\PythonWork\4.opencv\facilities_img.find_seg_point\facilities\save/' + '%s.png' % os.listdir(im1_path)[i], img3)
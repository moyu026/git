
import cv2
import numpy as np

img1_path = r'F:\PythonWork\3.opencv\4.electircal cabinet\feature_match\img\110.jpg'  # 待矫正图片
img2_path = r'F:\PythonWork\3.opencv\4.electircal cabinet\feature_match\img\112.jpg'
image1 = cv2.imread(img1_path)
image2 = cv2.imread(img2_path)

# SURF 特征
# surf = cv2.xfeatures2d.SURF_create() # opencv3.4,python3.6
surf = cv2.SIFT_create()
keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
keypoints2, descriptors2 = surf.detectAndCompute(image2, None)

# FLANN 匹配
flann = cv2.FlannBasedMatcher_create()
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 去掉最近邻m和次近邻n的距离比值太大的匹配结果
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

#  计算转换矩阵
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

# 透视变换
aligned_image = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))

cv2.imwrite('matched21.jpg', aligned_image)
# cv2.imshow('Aligned Image', aligned_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
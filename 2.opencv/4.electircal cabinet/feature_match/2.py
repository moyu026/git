import cv2
import numpy as np

# Load the two images
img1_path = r'F:\PythonWork\3.opencv\4.electircal cabinet\feature_match\img\1.jpg'
img2_path = r'F:\PythonWork\3.opencv\4.electircal cabinet\feature_match\img\2.jpg'
image1 = cv2.imread(img1_path)
image2 = cv2.imread(img2_path)

# Create SURF objects
surf = cv2.xfeatures2d.SURF_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
keypoints2, descriptors2 = surf.detectAndCompute(image2, None)

# Create a FLANN matcher
flann = cv2.FlannBasedMatcher_create()

# Match descriptors between the two images
'''
match: DMatch类型
queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
distance：代表这一组匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
其中 m 表示第一个最近的匹配，n 表示第二个最近的匹配。'''
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Filter good matches using the Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# # Extract matched keypoints
# src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
# dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#
# # Find the Homography matrix
# H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)
#
# # Warp image1 to align with image2
# aligned_image = cv2.warpPerspective(image1, H, (image2.shape[1], image2.shape[0]))
#
# # Display the aligned image
# cv2.imshow('Aligned Image', aligned_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

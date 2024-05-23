import cv2
import numpy as np

# 加载两张图像
image01 = cv2.imread(r"F:\PythonWork\3.opencv\4.electircal cabinet\feature_match\img\1.jpg", 1)
image02 = cv2.imread(r"F:\PythonWork\3.opencv\4.electircal cabinet\feature_match\img\2.jpg", 1)


# 灰度图转换
image1 = cv2.cvtColor(image01, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image02, cv2.COLOR_BGR2GRAY)

# 提取特征点
surf = cv2.xfeatures2d.SURF_create(400)  # 海塞矩阵阈值，调整精度
keyPoints1, descriptors1 = surf.detectAndCompute(image1, None)

# for kp in keyPoints1:
#     x, y = kp.pt
#     print(f"Keypoint at (x, y): ({x}, {y})")

print(descriptors1)




# keyPoints2, descriptors2 = surf.detectAndCompute(image2, None)
#
# # 使用FLANN匹配器进行特征点匹配
# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(descriptors1, descriptors2, k=2)
#
# # 提取最佳匹配
# good_matches = []
# for m, n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)
#
# # 绘制匹配结果
# img_match = cv2.drawMatches(image01, keyPoints1, image02, keyPoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.namedWindow("match", 0)
# cv2.imshow("match", img_match)
# cv2.imwrite("match1.jpg", img_match)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

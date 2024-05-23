import cv2
import numpy as np

# 加载两张图像
image01 = cv2.imread(r"F:\PythonWork\3.opencv\4.electircal cabinet\feature_match\img\1.jpg", 1)
image02 = cv2.imread(r"F:\PythonWork\3.opencv\4.electircal cabinet\feature_match\img\2.jpg", 1)

# 创建窗口并显示图像
# cv2.namedWindow("p2", 0)
# cv2.namedWindow("p1", 0)
# cv2.imshow("p2", image01)
# cv2.imshow("p1", image02)

# 灰度图转换
image1 = cv2.cvtColor(image01, cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image02, cv2.COLOR_BGR2GRAY)

# 提取特征点
'''
keyPoints1的属性
pt：即点，包含图像中关键点的位置x，y
angle：表示特征的方向，如前面处理过的图像中的径向线的方向。如果检测关键点的算法能够找到它，则设置它，否则设置为-1
size：表示特征的直径
response：表示关键点的强度，可对关键点排序或者滤除强度较弱的关键点等等
octave：表示发现该特征的图像金字塔层。广泛用于在检测关键点以及使用关键点进一步检测图像上可能大小不同的对象时，实现尺度独立，或者说尺度不变性。为了实现该功能，用相同的算法处理同一图像的不同尺度版本（仅缩小版本），称每一个尺度为金字塔的一个分组或一层。
class_id：为一个关键点或者一组关键点分配自定义的标识符
'''
# surf = cv2.xfeatures2d.SURF_create(3000)  # 海塞矩阵阈值，调整精度
surf = cv2.SIFT_create()
keyPoints1, descriptors1 = surf.detectAndCompute(image1, None)
keyPoints2, descriptors2 = surf.detectAndCompute(image2, None)

# 使用FLANN匹配器进行特征点匹配
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 提取最佳匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
img_match = cv2.drawMatches(image01, keyPoints1, image02, keyPoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.namedWindow("match", 0)
cv2.imshow("match", img_match)
# cv2.imwrite("match1.jpg", img_match)

cv2.waitKey(0)
cv2.destroyAllWindows()

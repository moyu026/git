import cv2

img_path = '../pointer_detect/pointer_img/10_0.jpg'
img = cv2.imread(img_path)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)  # 高斯模糊去噪（设定卷积核大小影响效果）

_, RedThresh1 = cv2.threshold(gray_img, 60, 80, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
_, RedThresh = cv2.threshold(blurred, 60, 100, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
th2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)  # 闭运算（链接块）
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 开运算（去噪点）

contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

# cv2.imshow('gray', gray_img)
cv2.imshow('blur', blurred)
# cv2.imshow('threshold', RedThresh)
# cv2.imshow('close', closed)
cv2.imshow('open', opened)
# cv2.imshow('threshold2', th2)
# cv2.imshow('threshold3', th3)
cv2.imshow('counters', img)
cv2.waitKey()

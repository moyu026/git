import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'F:\PythonWork\3.opencv\4.electircal cabinet\img_test\angle.jpg')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, gray = cv2.threshold(gray, 40, 80, cv2.THRESH_BINARY)

# 边缘检测
edges = cv2.Canny(gray, 180, 200, apertureSize=3)

kernel1 = np.ones((3, 3), np.uint8)
kernel2 = np.ones((3, 3), np.uint8)
# # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)
#
edges = cv2.dilate(edges, kernel1, iterations=2)
edges = cv2.erode(edges, kernel2, iterations=3)


# cv2.imshow('Detected Lines', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 霍夫变换
# lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=30)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=50)
# print(lines)

# 绘制检测到的直线

for x1, y1, x2, y2 in lines[0]:
    print('x1:', x1, 'y1:', y1, 'x2:', x2, 'y2:', y2)
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # 计算斜率
    k = (y2 - y1) / (x2 - x1)
    # 求反正切，再将得到的弧度转换为度
    result = np.arctan(k) * (180 / np.pi)
    print("直线倾斜角度为：" + str(result) + "度")


# # 显示结果
# cv2.imshow('line.jpg', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

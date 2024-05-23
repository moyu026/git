# coding=utf-8
import cv2
import numpy as np

# 读取输入图片
img0 = cv2.imread(r"F:\PythonWork\3.opencv\4.electircal cabinet\pointer_detect\result\save_folder\4.jpg")
# 将彩色图片转换为灰度图片
img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# 创建一个LSD对象
fld = cv2.ximgproc.createFastLineDetector(_length_threshold=80)

# 执行检测结果
dlines = fld.detect(img)

angle_list = []
for dline in dlines:
    x0 = int(round(dline[0][0]))
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))

    #计算长度
    length = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    print('长度', length)

    #计算角度
    if x0 != x1:
        k = (y0 - y1) / (x0 - x1)
        # 求反正切，再将得到的弧度转换为度
        angle = np.arctan(k) * (180 / np.pi)
        print('角度', angle)

    cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)

    if angle < 89 and angle > 10:
        angle_list.append(angle)
        cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)
    average_angle = np.mean(angle_list)

# print(angle_list)
print(average_angle)
# 显示并保存结果
# # cv2.imwrite('fld.jpg', img0)
cv2.imshow("LSD", img0)
cv2.waitKey(0)

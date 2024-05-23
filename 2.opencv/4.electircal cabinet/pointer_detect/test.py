# coding=utf-8
import os

import cv2
import numpy as np

img_folder = '../pointer_detect/pointer'
save_folder = '../pointer_detect/result/save_folder'

img_list = os.listdir(img_folder)
for imgs in img_list:
    img_path = os.path.join(img_folder, imgs)
    save_img_path = os.path.join(save_folder, imgs)
    # 读取输入图片
    img0 = cv2.imread(img_path)
    # 将彩色图片转换为灰度图片
    img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

    # 创建一个LSD对象
    fld = cv2.ximgproc.createFastLineDetector(_length_threshold=50)

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
        k = (y0 - y1) / (x0 - x1)
        # 求反正切，再将得到的弧度转换为度
        angle = np.arctan(k) * (180 / np.pi)
        print('角度', angle)

        # cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)

        if angle < 80 and angle > 10:
            angle_list.append(angle)
            cv2.line(img0, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)
        # average_angle = np.mean(angle_list)

    cv2.imwrite(save_img_path, img0)

# print(angle_list)
# # print(average_angle)
# # 显示并保存结果
# # # cv2.imwrite('fld.jpg', img0)
# cv2.imshow("LSD", img0)
# cv2.waitKey(0)

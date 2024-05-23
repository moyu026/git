import cv2
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

img_folder = r'F:\PythonWork\dataset\firedataset\fire_bg\img_black'
save_folder = r'F:\PythonWork\dataset\firedataset\fire_bg\img_white'
txt_folder = r'F:\PythonWork\dataset\firedataset\fire_bg\txt_point'
img_list = os.listdir(img_folder)

for imgs in img_list:
    img_path = os.path.join(img_folder, imgs)
    save_path = os.path.join(save_folder, imgs)
    txt_path = os.path.join(txt_folder, imgs)[:-3] + 'txt'

    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
    ret, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)  # 二值化， ret:设定的阈值， binary：二值化后的图像
    kernel = np.ones((1, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, anchor=(2, 0), iterations=6) # cv2.MORPH_CLOSE 进行闭运算， 指的是先进行膨胀操作，再进行腐蚀操作
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 检测轮廓，contours：轮廓坐标 ，hierarchy：每条轮廓对应的属性

    max_contours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 10000:
            max_contours.append(i)
    pts = max_contours

    with open(txt_path, 'w') as f:
        f.write(str(pts))
        print(txt_path)

    # 和原始图像一样大小的0矩阵，作为mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # 在mask上将多边形区域填充为白色
    cv2.polylines(mask, pts, 1, 255)  # 描绘边缘
    cv2.fillPoly(mask, pts, 255)  # 填充
    # 逐位与，得到裁剪后图像，此时是黑色背景
    dst = cv2.bitwise_and(img, img, mask=mask)
    # 添加白色背景
    bg = np.ones_like(img, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)  # bg的多边形区域为0，背景区域为255
    dst_white = bg + dst
    # cv2.imwrite(save_path, dst_white)







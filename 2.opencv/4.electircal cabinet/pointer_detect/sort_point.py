import json
import cv2
import math
import numpy as np

def sort_point(points):
    # 比较纵坐标，选出较大的两个点作为候选
    top_two = sorted(points, key=lambda p: p[1], reverse=True)[:2]

    # 在纵坐标较大的两个点中比较横坐标，选择横坐标较大的一个为右下，另一个为左下
    right_bottom, left_bottom = sorted(top_two, key=lambda p: p[0], reverse=True)

    # 纵坐标中较小的两个点
    bottom_two = sorted(points, key=lambda p: p[1], reverse=False)[:2]

    right_top, left_top = sorted(bottom_two, key=lambda p: p[0], reverse=True)

    result_point = [right_top, left_top, left_bottom, right_bottom]  # 返回的点的位置顺序不变
    return result_point

file_path = '../pointer_detect/pointer_img/12.json'
img_path = '../pointer_detect/pointer_img/12.JPG'

img = cv2.imread(img_path)
drawimg = img.copy()
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = np.zeros(img.shape[0:2], dtype=np.uint8)

bg = np.zeros(img.shape, np.uint8)
# 使用白色填充图片区域,默认为黑色
# bg.fill(255)

with open(file_path, 'r') as load_f:
    json_dict = json.load(load_f)
    indx = 0
    for obj_dict in json_dict['objects']:
        pts = obj_dict['segmentation']

        contours = ((np.array(pts)).astype(int)).reshape((-1, 1, 2))

        epsilon = 0.01 * cv2.arcLength(contours, True)
        # 预测多边形
        box = cv2.approxPolyDP(contours, epsilon, True)

        box_list = sort_point(box.reshape(-1, 2).tolist())

        box = (np.array(box_list)).reshape((-1, 1, 2))
        # print(box_list)
        draw_img = cv2.polylines(drawimg, [box], True, (0, 0, 255), 2)

        orignal_W = math.ceil(np.sqrt((box[3][0][1] - box[2][0][1]) ** 2 + (box[3][0][0] - box[2][0][0]) ** 2))  # 两个点坐标之间的距离
        orignal_H = math.ceil(np.sqrt((box[3][0][1] - box[0][0][1]) ** 2 + (box[3][0][0] - box[0][0][0]) ** 2))

        pt1 = np.float32([box[3], box[2], box[1], box[0]])  # 按照右下，左下，左上，右上的顺序
        pt2 = np.float32([[int(orignal_W + 1), int(orignal_H + 1)], [0, int(orignal_H + 1)], [0, 0], [int(orignal_W + 1), 0]])

        M = cv2.getPerspectiveTransform(pt1, pt2)
        result_img = cv2.warpPerspective(img, M, (int(orignal_W + 3), int(orignal_H + 1)))

        img_name = str(indx) + '.jpg'

        cv2.imwrite(img_name, result_img)

        indx += 1

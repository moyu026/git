import math
import os
import cv2
import numpy
import numpy as np
import torch
from skimage import morphology


class MeterReader(object):

    def __init__(self):
        pass

    def __call__(self, image, point_mask, dail_mask, word_mask, number, std_point, number_boxes):

        img_result = image.copy()
        value = self.find_lines(img_result, point_mask, dail_mask, number, std_point, number_boxes)
        print("value", value)

        return value

    def find_lines(self, ori_img, pointer_mask, dail_mask, number, std_point, number_boxes):

        # 实施骨架算法
        pointer_skeleton = morphology.skeletonize(pointer_mask)
        pointer_edges = pointer_skeleton * 255
        pointer_edges = pointer_edges.astype(np.uint8)
        # cv2.imshow("pointer_edges", pointer_edges)
        # cv2.waitKey(0)

        dail_mask = np.clip(dail_mask, 0, 1)
        dail_edges = dail_mask * 255
        dail_edges = dail_edges.astype(np.uint8)
        # cv2.imshow("dail_edges", dail_edges)
        # cv2.waitKey(0)

        pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10,
                                        maxLineGap=400)
        coin1, coin2 = None, None

        try:
            for x1, y1, x2, y2 in pointer_lines[0]:
                coin1 = (x1, y1)
                coin2 = (x2, y2)
                # cv2.line(ori_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                # cv2.imshow('pointer', ori_img)
                # cv2.waitKey()
        except TypeError:
            return "can not detect pointer"

        # 计算刻度零点所在水平直线与指针所在直线的交点
        origin = std_point[0]
        origin_right = (std_point[0][0] + 10, std_point[0][1])

        intresection_point = self.find_intersection(origin, origin_right, coin1, coin2)

        if intresection_point == None:
            return 0
        else:

            line_points = [intresection_point, coin2, coin1]
            min_point, max_point = self.find_max_min_y(line_points)

            pointer_line = (max_point, min_point)

            # # 由计算中心点与指针两端点的距离比较改为右下角点与指针端点的距离比较
            # h, w, _ = ori_img.shape
            # # center = (0.5 * w, 0.5 * h)
            # center = (w, h)
            # dis1 = (coin1[0] - center[0]) ** 2 + (coin1[1] - center[1]) ** 2
            # dis2 = (coin2[0] - center[1]) ** 2 + (coin2[1] - center[1]) ** 2
            # if dis1 <= dis2:
            #     pointer_line = (coin1, coin2)
            # else:
            #     pointer_line = (coin2, coin1)
            #
            # # print("pointer_line", pointer_line)
            #
            # if std_point==None:
            #     return "can not detect dail"

            # calculate angle
            a1 = std_point[0]  # 刻度零点的坐标
            a2 = std_point[1]  # 第一个刻度值的坐标
            # cv2.circle(ori_img, a1, 2, (255, 0, 0), 2)
            # cv2.circle(ori_img, a2, 2, (255, 0, 0), 2)
            one = [[pointer_line[0][0], pointer_line[0][1]], [a1[0], a1[1]]]
            two = [[pointer_line[0][0], pointer_line[0][1]], [a2[0], a2[1]]]
            three = [[pointer_line[0][0], pointer_line[0][1]], [pointer_line[1][0], pointer_line[1][1]]]
            # print("one", one)
            # print("two", two)
            # print("three",three)

            one = np.array(one)
            two = np.array(two)
            three = np.array(three)

            v1 = one[1] - one[0]  # 指针中心端点与到刻度零点的向量
            v2 = two[1] - two[0]  # 指针中心端点到第一个刻度点的向量
            v3 = three[1] - three[0]  # 指针所在直线的向量

            distance = self.get_distance_point2line([a1[0], a1[1]], [pointer_line[0][0], pointer_line[0][1], pointer_line[1][0], pointer_line[1][1]])
            # print("dis",distance)

            flag = self.judge(pointer_line[0], std_point[0], pointer_line[1])
            # print("flag",flag)

            std_ang = self.angle(v1, v2)  #
            # print("std_result", std_ang)
            now_ang = self.angle(v1, v3)
            if flag > 0:
                now_ang = 360 - now_ang
            # print("now_result", now_ang)

            # calculate value
            if number != None and number[0] != "":
                two_value = float(number[0])
            else:
                return "can not recognize number"
            if std_ang * now_ang != 0:
                value = (two_value / std_ang)
                value = value * now_ang
            else:
                return "angle detect error"

            if flag > 0 and distance < 40:
                value = 0.00
            else:
                value = round(value, 3)

            # font = cv2.FONT_HERSHEY_SIMPLEX
            # ori_img = cv2.putText(ori_img, str(value), (30, 30), font, 1.2, (255, 0, 255), 2)
            #
            # x1, y1, x2, y2, x3, y3, x4, y4 = number_boxes[0]
            # pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            # pts = pts.reshape((-1, 1, 2))
            # cv2.polylines(ori_img, [pts], isClosed=True, color=(0, 0, 255), thickness=1)

            # cv2.imshow(str(number), ori_img)
            # cv2.waitKey(0)

            return value

    def get_distance_point2line(self, point, line):
        """
        Args: 计算点到直线距离
            point: [x0, y0]
            line: [x1, y1, x2, y2]
        """
        line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
        vec1 = line_point1 - point
        vec2 = line_point2 - point
        distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
        return distance

    def judge(self, p1, p2, p3):
        '''
        判断点p3在直线p1p2的哪一侧
            value>0, 点p3在直线上侧
            value<0, 点p3在直线下侧
            value=0，点p3在直线上
        '''
        A = p2[1] - p1[1]
        B = p1[0] - p2[0]
        C = p2[0] * p1[1] - p1[0] * p2[1]

        value = A * p3[0] + B * p3[1] + C

        return value

    def angle(self, v1, v2):
        '''
        计算两个向量之间的夹角
        '''
        lx = np.sqrt(v1.dot(v1))  # v1.dot(v1)等价于np.dot(v1,v1)
        ly = np.sqrt(v2.dot(v2))
        cos_angle = v1.dot(v2) / (lx * ly)

        angle = np.arccos(cos_angle)
        angle2 = angle * 360 / 2 / np.pi

        return angle2

    def find_intersection(self, point1_line1, point2_line1, point1_line2, point2_line2):  # 已知直线上的两个点坐标，计算两条直线的交点
        # 计算直线1的斜率和截距
        slope_line1 = (point2_line1[1] - point1_line1[1]) / (point2_line1[0] - point1_line1[0])
        intercept_line1 = point1_line1[1] - slope_line1 * point1_line1[0]

        # 计算直线2的斜率和截距
        slope_line2 = (point2_line2[1] - point1_line2[1]) / (point2_line2[0] - point1_line2[0])
        intercept_line2 = point1_line2[1] - slope_line2 * point1_line2[0]

        if slope_line1 - slope_line2 == 0:
            return None
        else:
            # 计算交点的x坐标
            x_intersection = (intercept_line2 - intercept_line1) / (slope_line1 - slope_line2)

            # 计算交点的y坐标
            y_intersection = slope_line1 * x_intersection + intercept_line1

            return (int(x_intersection), int(y_intersection))

    def find_max_min_y(self, points):  # 找出三个点中位于两端的点
        # 初始值设为第一个点
        max_point = points[0]
        min_point = points[0]

        # 遍历剩余的点
        for point in points[1:]:
            # 更新最大值和最小值
            if point[1] > max_point[1]:
                max_point = point
            elif point[1] < min_point[1]:
                min_point = point

        return min_point, max_point


if __name__ == '__main__':
    tester = MeterReader()
    root = 'data/images/val'
    for image_name in os.listdir(root):
        print(image_name)
        path = f'{root}/{image_name}'
        image = cv2.imread(path)
        result = tester(image)
        print(result)
        # cv2.imshow('a', image)
        # cv2.waitKey()

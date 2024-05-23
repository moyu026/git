import cv2
import math
import numpy as np

def Img_Outline(input_dir):
    original_img = cv2.imread(input_dir)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)                     # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv2.threshold(blurred, 40, 80, cv2.THRESH_BINARY)    # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))          # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)           # 开运算（去噪点）
    return original_img, gray_img, RedThresh, closed, opened


def findContours_img(original_img, opened):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[1]   # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)                                    # 获取包围盒（中心点，宽高，旋转角度）
    box = np.int0(cv2.boxPoints(rect))                           # box
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)

    print("box[0]:", box[0])
    print("box[1]:", box[1])
    print("box[2]:", box[2])
    print("box[3]:", box[3])
    return box, draw_img

def Perspective_transform(box,original_img):
    # 获取画框宽高(x=orignal_W,y=orignal_H)
    orignal_W = math.ceil(np.sqrt((box[3][1] - box[2][1])**2 + (box[3][0] - box[2][0])**2))
    orignal_H= math.ceil(np.sqrt((box[3][1] - box[0][1])**2 + (box[3][0] - box[0][0])**2))

    # 原图中的四个顶点,与变换矩阵
    pts1 = np.float32([box[3], box[0], box[1], box[2]])
    pts2 = np.float32([[int(orignal_W+1), int(orignal_H+1)], [0, int(orignal_H+1)], [0, 0], [int(orignal_W+1), 0]])

    # 生成透视变换矩阵；进行透视变换
    M = cv2.getPerspectiveTransform(pts1, pts2)
    result_img = cv2.warpPerspective(original_img, M, (int(orignal_W+3), int(orignal_H+1)))

    return result_img

def line_detect(line_img):

    # 转换为灰度图像
    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)

    ret, gray = cv2.threshold(gray, 40, 80, cv2.THRESH_BINARY)

    # 边缘检测
    edges = cv2.Canny(gray, 180, 200, apertureSize=3)

    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)

    edges = cv2.dilate(edges, kernel1, iterations=2)
    edges = cv2.erode(edges, kernel2, iterations=3)

    # 霍夫变换
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=30)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=50)
    # print(lines)

    # 绘制检测到的直线

    for x1, y1, x2, y2 in lines[0]:
        print('x1:', x1, 'y1:', y1, 'x2:', x2, 'y2:', y2)
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 计算斜率
        k = (y2 - y1) / (x2 - x1)
        # 求反正切，再将得到的弧度转换为度
        result = np.arctan(k) * (180 / np.pi)
        print("直线倾斜角度为：" + str(result) + "度")

    return line_img



if __name__=="__main__":
    input_dir = "../pointer_detect/pointer_img/10_0.jpg"
    original_img, gray_img, RedThresh, closed, opened = Img_Outline(input_dir)
    box, draw_img = findContours_img(original_img, opened)
    result_img = Perspective_transform(box, original_img)
    # line_img = line_detect(result_img)
    cv2.imshow("original", original_img)
    # cv2.imshow("gray", gray_img)
    # cv2.imshow("closed", closed)
    # cv2.imshow("opened", opened)
    cv2.imshow("draw_img", draw_img)
    cv2.imshow("result_img", result_img)
    cv2.imwrite('result.jpg', result_img)
    # cv2.imshow("line_img", line_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

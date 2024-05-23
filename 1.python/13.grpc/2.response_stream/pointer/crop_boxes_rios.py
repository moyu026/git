import cv2
import json
import os

import numpy as np
import cv2
import torch


def param2theta(param, w, h):
    param = np.vstack([param, [0, 0, 1]])
    param = np.linalg.inv(param)  # 计算矩阵的逆矩阵

    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0]
    theta[0, 1] = param[0, 1] * h / w
    theta[0, 2] = param[0, 2] * 2 / w + theta[0, 0] + theta[0, 1] - 1
    theta[1, 0] = param[1, 0] * w / h
    theta[1, 1] = param[1, 1]
    theta[1, 2] = param[1, 2] * 2 / h + theta[1, 0] + theta[1, 1] - 1
    return theta


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0, 1] != leftMost[1, 1]:
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    else:
        leftMost = leftMost[np.argsort(leftMost[:, 0])[::-1], :]
    (tl, bl) = leftMost
    if rightMost[0, 1] != rightMost[1, 1]:
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    else:
        rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
    (tr, br) = rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return tl, tr, br, bl


img_dir = r'F:\PythonWork\1.detection\5.eletrical_cabinet\pointer\Detect-and-read-meters-2\data\images'
img_list = os.listdir(img_dir)
json_dir = r'F:\PythonWork\1.detection\5.eletrical_cabinet\pointer\Detect-and-read-meters-2\data\annotations\train1'
save_img_dir = r'F:\PythonWork\0.other\label_test\file\1\save_roi'

for imgs in img_list:
    img_path = os.path.join(img_dir, imgs)
    json_path = json_dir + '\\' + imgs[:-3] + 'json'

    image = cv2.imread(img_path)
    resize_h, resize_w = (100, 180)

    width = image.shape[0]
    height = image.shape[1]

    mapped_x1, mapped_y1 = (0, 0)
    mapped_x4, mapped_y4 = (0, resize_h)

    mapped_x2, mapped_y2 = (resize_w, 0)

    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    info = load_dict['shapes']
    for i in range(len(info)):
        points = info[i]['points']
        points = np.array(points).astype(np.int32)
        label = info[i]['label']

        tl, tr, br, bl = order_points(points)

        src_pts = np.float32([tl, tr, bl])
        dst_pts = np.float32([
            (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
        ])

        matrix = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))  # 仿射变换
        matrix_result = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))  # 原图经过仿射矩阵M变换后的结果1

        crop_img = matrix_result[0:resize_h, 0:resize_w]

        matrix_1 = param2theta(matrix, width, height)
        # matrix_1_result = cv2.warpAffine(matrix_result, matrix_1, (image.shape[1], image.shape[0]))  # 仿射变换后的结果1经过仿射矩阵M的逆矩阵调整后的矩阵M'变换后的结果2

        matrix_1 *= 1e20  # 避免类型转换过程中出现的数值精度损失
        matrix_1 = torch.tensor(matrix_1, dtype=torch.float)
        matrix_1 /= 1e20

        feature = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()
        grid = torch.nn.functional.affine_grid(matrix_1.unsqueeze(0), feature.size())  # pytorch 中的仿射变换
        feature_rotated = torch.nn.functional.grid_sample(feature, grid)

        feature_rotated_crop = feature_rotated[:, :, 0:resize_h, 0:resize_w]
        feature_rotated_crop_img = feature_rotated_crop.squeeze().numpy().transpose(1, 2, 0).astype(np.uint8)

        roi_name = save_img_dir + '\\' + imgs[:-4] + '-' + label + '.jpg'

        cv2.imwrite(roi_name, crop_img)

        print(json_path)

# # 显示原始图像和变换后的图像
# cv2.imshow('../file/1/save_transform/Original.jpg', image)
# cv2.imshow('../file/1/save_transform/matrix_result.jpg', matrix_result)
# # cv2.imshow('../file/1/save_transform/matrix_1_result.jpg', matrix_1_result)
# # cv2.imshow('../file/1/save_transform/feature_rotated_img.jpg', feature_rotated_np)
# # cv2.imshow('../file/1/save_transform/feature_rotated_crop.jpg', feature_rotated_crop_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2 as cv
import os
import numpy as np

img_point_folder = 'img_2/pts_threshold'
img_point_list = os.listdir(img_point_folder)

img_model_folder = 'img_2/obj_threshold'
img_model_list = os.listdir(img_model_folder)


count = 0
for img_models in img_model_list:
    img_model_path = os.path.join(img_model_folder, img_models)

    img_model = cv.imread(img_model_path)
    img_model = img_model[185:910, 420:960]

    gray = cv.cvtColor(img_model, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 254, 255, cv.THRESH_BINARY)  # ret:设定的阈值， binary：二值化后的图像
    kernel = np.ones((1, 5), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    x = []
    y = []

    for i in range(len(contours[0])):
        x.append(contours[0][i][0][0])
        y.append(contours[0][i][0][1])
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    img_model = img_model[min_y - 10:max_y + 10, min_x - 10:max_x + 10]

    theight, twidth = img_model.shape[:2]

    # cv.imshow('img', img_model)
    # cv.waitKey(0)

    img_name = []
    mini = []
    result = {}

    for img_points in img_point_list:

        img_point_path = os.path.join(img_point_folder, img_points)

        img_point = cv.imread(img_point_path)
        img_point = img_point[85:910, 320:1760]

        # cv.imshow(img_point_path, img_point)
        # cv.waitKey(0)

        result = cv.matchTemplate(img_point, img_model, cv.TM_SQDIFF_NORMED)

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        strmin_val = str(min_val)

        cv.rectangle(img_point, min_loc, (min_loc[0] + twidth, min_loc[1] + theight), (0, 0, 225), 2)

        # cv.imshow("MatchResult----MatchingValue=" + strmin_val, img_point)
        # cv.waitKey(0)

        mini.append(min_val)
        img_name.append(img_points)

    result = dict(zip(img_name, mini))
    best_result = min(result.items(), key=lambda x: x[1])

    name = list(enumerate(result))

    if best_result[0][0] != img_models[0]:
        print('=====================')
        print(result)
        print(best_result)
        print(img_models)
    else:
        print(result)
        print(best_result)
        print(img_models)


    # print('indx_name:', name[0][1], result[name[indx][1]])
    # print('best_name:', best_result[0])
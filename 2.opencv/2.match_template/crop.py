import cv2 as cv
import os
import numpy as np

img_point_folder = 'img1/PTS/top'
img_point_list = os.listdir(img_point_folder)
img_point_save_folder = r'img1/PTS/pts_crop/top'

img_model_folder = 'img1/OBJ/top'
img_model_list = os.listdir(img_model_folder)
img_model_save_folder = r'img1/OBJ/obj_crop/top'

for img_points in img_point_list:
    img_point_path = os.path.join(img_point_folder, img_points)

    img_point_save_path = os.path.join(img_point_save_folder, img_points)

    img_point = cv.imread(img_point_path)
    img_point = img_point[250:760, 520:940]

    cv.imwrite(img_point_save_path, img_point)


for img_models in img_model_list:
    img_model_path = os.path.join(img_model_folder, img_models)

    img_model_save_path = os.path.join(img_model_save_folder, img_models)

    img_model = cv.imread(img_model_path)
    img_model = img_model[85:910, 320:1760]

    cv.imwrite(img_model_save_path, img_model)


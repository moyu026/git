import json
import cv2
import numpy as np

file_path = 'file/2.json'
img_path = 'file/2.jpg'

img = cv2.imread(img_path)

# bbox
# with open(file_path, 'r') as load_f:
#     json_dict = json.load(load_f)
#     for obj_dict in json_dict['objects']:
#         print(obj_dict['bbox'])
#         xmin, ymin, xmax, ymax = obj_dict['bbox'][0], obj_dict['bbox'][1], obj_dict['bbox'][2], obj_dict['bbox'][3]
#         cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), thickness=2)
#
# cv2.imwrite('../HSV/file/bbx.jpg', img)

# segment point
# with open(file_path, 'r') as load_f:
#     json_dict = json.load(load_f)
#     for obj_dict in json_dict['objects']:
#         pts = obj_dict['segmentation']
#         pts = np.array(pts)
#         pts = pts.astype(int)
#         pts = pts.reshape((-1, 1, 2))
#         cv2.polylines(img, [pts], True, (0, 0, 255), 2)
# cv2.imshow('j', img)
# cv2.waitKey()

with open(file_path, 'r') as load_f:
    json_dict = json.load(load_f)
    for obj_dict in json_dict['objects']:
        xmin, ymin, xmax, ymax = obj_dict['bbox'][0], obj_dict['bbox'][1], obj_dict['bbox'][2], obj_dict['bbox'][3]
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), thickness=1)

        pts = obj_dict['segmentation']
        pts = np.array(pts)
        pts = pts.astype(int)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 255, 0), 1)

cv2.imshow('j', img)
cv2.waitKey()




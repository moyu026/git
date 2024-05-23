from ultralytics import YOLO
import numpy as np
import os
import cv2
from skimage import morphology

path = r'demo/1/det_777.jpg'
# Load a model
model = YOLO(r'trian_weights/v8_segmeny/best.pt')  # load an official model
# results = model.predict(path, save=True, conf=0.5,  save_txt=True)
results = model.predict(path, conf=0.5)
ori_img = results[0].orig_img
img_shape = results[0].orig_shape
mask_640 = results[0].masks.data.cpu().numpy().transpose(1, 2, 0)

img_mask = cv2.resize(mask_640, (img_shape[1], img_shape[0]))
# cv2.imwrite('img_msk.jpg', img_mask*255)
# 骨架算法
pointer_skeleton = morphology.skeletonize(img_mask)
pointer_edges = pointer_skeleton * 255
pointer_edges = pointer_edges.astype(np.uint8)

pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10, maxLineGap=400)
for x1, y1, x2, y2 in pointer_lines[0]:
    cv2.line(ori_img, (x1, y1), (x2, y2), (255, 0, 255), 2)

cv2.imwrite('pointer.jpg', ori_img)



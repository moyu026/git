import json
import cv2
import numpy as np

file_path = 'file/1-1.json'
img_path = 'file/1-1.jpg'

lower_bound = np.array([40, 80, 80])
upper_bound = np.array([150, 255, 255])

img = cv2.imread(img_path)
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask = np.zeros(img.shape[0:2], dtype=np.uint8)

with open(file_path, 'r') as load_f:
    json_dict = json.load(load_f)
    for obj_dict in json_dict['objects']:
        pts = obj_dict['segmentation']
        pts = ((np.array(pts)).astype(int)).reshape((-1, 1, 2))
        # cv2.polylines(img, [pts], True, (255, 0, 0), 1)
        cv2.fillPoly(mask, [pts], color=255)

        hsv_mean = cv2.mean(hsv_image, mask=mask)

        print(f'Mean Hue (色相): {hsv_mean[0]}')
        print(f'Mean Saturation (饱和度): {hsv_mean[1]}')
        print(f'Mean Value (明度): {hsv_mean[2]}')

        if (hsv_mean[0] >= lower_bound[0] and hsv_mean[1] >= lower_bound[1] and hsv_mean[2] >= lower_bound[2] and
                hsv_mean[0] <= upper_bound[0] and hsv_mean[1] <= upper_bound[1] and hsv_mean[2] <= upper_bound[2]):
            print("HSV均值在指定范围内")
        else:
            print("HSV均值不在指定范围内")

cv2.imwrite('11.jpg', img)
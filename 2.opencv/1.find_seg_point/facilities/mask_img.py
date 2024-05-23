import os
import cv2

img_folder = r'F:\PythonWork\4.opencv\1.find_seg_point\facilities\facilities_img\val'
mask_folder = r'F:\PythonWork\4.opencv\1.find_seg_point\facilities\mask\val'
save_folder = r'F:\PythonWork\4.opencv\1.find_seg_point\facilities\save\val'

img_list = os.listdir(img_folder)
mask_list = os.listdir(mask_folder)

for i in range(len(img_list)):
    img_path = os.path.join(img_folder, img_list[i])
    mask_path = os.path.join(mask_folder, mask_list[i])
    save_path = os.path.join(save_folder, img_list[i])

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # cv2.imshow('img', img)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    dst = cv2.bitwise_and(img, img, mask=mask)

    cv2.imwrite(save_path, dst)
    print(save_path)




import cv2
import os

img_folder = r'F:\PythonWork\ocr\PaddleOCR-release-2.7\StyleText\img'
save_folder = r'F:\PythonWork\ocr\PaddleOCR-release-2.7\StyleText\img\resize'
img_list = os.listdir(img_folder)

for imgs in img_list:
    img_path = os.path.join(img_folder, imgs)
    save_img_path = os.path.join(save_folder, imgs)
    img = cv2.imread(img_path)

    h, w, _ = img.shape
    # print(h, w)
    w1 = int(w/(h/32))


    img_resize = cv2.resize(img, (w1, 32))

    cv2.imwrite(save_img_path, img_resize)

    print(save_img_path)

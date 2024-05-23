import cv2
import os

img_folder = r'F:\PythonWork\dataset\ocr\self_dataset\0.train_dataset\paddle_gen1\val'
save_folder = r'F:\PythonWork\dataset\ocr\self_dataset\1.gan\gray_gen1\val'

for imgs in os.listdir(img_folder):
    img_path = os.path.join(img_folder, imgs)
    save_path = os.path.join(save_folder, imgs)

    # 读取二值化图像
    binary_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 转换为灰度图像
    gray_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(save_path, gray_image)
    print(save_path)


# # 显示灰度图像
# cv2.imshow('Gray Image', gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

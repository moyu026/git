import cv2
import os


img_path = r'img/fakes001040.png'
save_path = r'img/little_imgs'

img = cv2.imread(img_path)

i = 0   # 29
j = 0   # 15
for i in range(29):
    for j in range(15):
        img1 = img[256*i: 256*(i+1), 256*j: 256*(j+1)]
        img_little_name = str(i) + str(j) + '.jpg'
        img_little_path = os.path.join(save_path, img_little_name)
        cv2.imwrite(img_little_path, img1)

        print(img_little_path)



# cv2.imshow('img1', img1)
# cv2.waitKey(0)
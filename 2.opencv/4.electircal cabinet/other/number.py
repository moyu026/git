import cv2

img_path = '..//img/number/number.png'

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 150, 180, cv2.THRESH_BINARY) # ret:设定的阈值， binary：二值化后的图像
cv2.imshow("img", binary)
cv2.waitKey(0)
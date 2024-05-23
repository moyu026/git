import cv2 as cv

image = cv.imread('img_test/angle.jpg')

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

ret, img_threshold = cv.threshold(gray, 40, 60, cv.THRESH_BINARY)
# img_medianBlur = cv.medianBlur(img, 5)

# cv.imwrite('threshold.jpg', img_threshold)


# cv.imshow('img.png', img)
cv.imshow('img_blur.png', img_threshold)
cv.waitKey(0)

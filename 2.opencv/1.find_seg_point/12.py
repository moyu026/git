import cv2


img1 = cv2.imread('background/facilities_img.jpg')
img2 = cv2.imread('img/facilities_img.jpeg')


img2 = cv2.resize(img2, (300, 300))
img1 = cv2.resize(img1, (460, 460))

# I want to put logo on top-left corner, So I create a ROI
# 首先获取原始图像roi
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

# 原始图像转化为灰度值
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# cv2.imshow('img2gray', img2gray)
# cv2.waitKey(0)
'''
将一个灰色的图片，变成要么是白色要么就是黑色。（大于规定thresh值就是设置的最大值（常为255，也就是白色））
'''
# 将灰度值二值化，得到ROI区域掩模
ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)

cv2.imshow('mask', mask)
cv2.imwrite('mask2.jpg', mask)
cv2.waitKey(0)

# ROI掩模区域反向掩模
mask_inv = cv2.bitwise_not(mask)

cv2.imshow('mask_inv', mask_inv)
cv2.waitKey(0)

# 掩模显示背景
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

cv2.imshow('img1_bg', img1_bg)
# cv2.waitKey(0)

# 掩模显示前景
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
cv2.imshow('img2_fg', img2_fg)
# cv2.waitKey(0)

# 前背景图像叠加
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
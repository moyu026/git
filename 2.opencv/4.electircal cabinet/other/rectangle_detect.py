import cv2

image = cv2.imread('img/8.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 150, 205)
# edges = cv2.resize(edges, (800, 1000), interpolation=cv2.INTER_LINEAR)

cv2.imwrite('output/test/canny.jpg', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# cv2.imshow('Rectangles Detected', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

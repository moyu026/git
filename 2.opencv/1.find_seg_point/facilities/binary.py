import cv2


# 创建一个空函数，稍后将用作滑动条的回调函数
def nothing(x):
    pass

# 加载图像
img_path = r'F:\PythonWork\4.opencv\1.find_seg_point\facilities\depth_img\val\160-dpt_beit_large_512.png'
image = cv2.imread(img_path, 0)  # 这里假设你的图像文件名为image.jpg
image = cv2.resize(image, (640, 640))

# 创建窗口和滑动条
cv2.namedWindow('Threshold')
cv2.createTrackbar('Threshold Value', 'Threshold', 0, 255, nothing)

while True:
    # 获取滑动条的当前位置
    threshold_value = cv2.getTrackbarPos('Threshold Value', 'Threshold')

    # 对图像进行二值化
    _, thresholded_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # 显示二值化图像
    cv2.imshow('Threshold', thresholded_image)

    # 按下 ESC 键退出循环
    if cv2.waitKey(1) == 27:
        break

# 清理资源并关闭窗口
cv2.destroyAllWindows()

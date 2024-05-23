import cv2

# 打开图像文件
image = cv2.imread("img_test/1.jpg")  # 替换为你的图像文件路径

# 获取图像的高度和宽度
height, width, channels = image.shape

# 定义行和列的数量
rows = 3
cols = 3

# 计算每个小区域的宽度和高度
region_width = width // cols
region_height = height // rows

# 循环遍历每个小区域
for i in range(rows):
    for j in range(cols):
        # 计算当前小区域的左上角和右下角坐标
        left = j * region_width
        upper = i * region_height
        right = (j + 1) * region_width
        lower = (i + 1) * region_height

        # 切割并保存当前小区域
        region = image[upper:lower, left:right]
        cv2.rectangle(image, (left, upper), (right, lower), (0, 0, 255), thickness=2)

cv2.imshow('region', image)


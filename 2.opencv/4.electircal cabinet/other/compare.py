from PIL import Image, ImageChops

# 读取两张图片
image1 = Image.open("img_test/5.jpg")
image2 = Image.open("img_test/5-1.jpg")

# 获取图片的尺寸
width, height = image1.size

# 比较两张图片
diff_image = ImageChops.difference(image1, image2)

# 转换为灰度图像
diff_image = diff_image.convert("L")

# 设置阈值，将不一致的像素标记为白色
threshold = 30
diff_image = diff_image.point(lambda p: 255 if p > threshold else 0)

# 获取不一致的像素坐标
diff_coordinates = []
for y in range(height):
    for x in range(width):
        if diff_image.getpixel((x, y)) == 255:
            diff_coordinates.append((x, y))

# 输出不一致的像素坐标
for coordinate in diff_coordinates:
    print("Different pixel at coordinates:", coordinate)

# 显示不一致的像素图像
diff_image.show()

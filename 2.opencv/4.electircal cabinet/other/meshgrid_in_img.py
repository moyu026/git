import numpy as np
from PIL import Image, ImageDraw


img_path = r'img_test/2.jpg'
image = Image.open(img_path)
draw = ImageDraw.Draw(image)

image_width, image_height = image.size
num_columns = 11
num_rows = 7

# 生成网格的x和y坐标
x = np.arange(300, image_width, 400)
y = np.arange(550, image_height, 450)

# 创建网格点的坐标
xv, yv = np.meshgrid(x, y)

# print(yv)

# 绘制网格点并连接成红色线条
for i in range(xv.shape[0]):
    for j in range(xv.shape[1]):
        point_x = xv[i, j]
        point_y = yv[i, j]

        # 绘制竖线
        if j > 0:
            prev_point_x = xv[i, j - 1]
            prev_point_y = yv[i, j - 1]
            draw.line([(prev_point_x, prev_point_y), (point_x, point_y)], fill="red", width=5)

        # 绘制横线
        if i > 0:
            prev_point_x = xv[i - 1, j]
            prev_point_y = yv[i - 1, j]
            draw.line([(prev_point_x, prev_point_y), (point_x, point_y)], fill="red", width=5)
# 保存图像或显示图像
image.save("grid_lines_image.png")
image.show()
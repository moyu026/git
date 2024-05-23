from PIL import Image, ImageDraw

def find_green_pixels(image, range_value):
    green_pixels = []

    width, height = image.size

    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))
            if is_green_color(pixel, range_value):
                green_pixels.append((x, y))

    return green_pixels

def is_green_color(rgb, range_value):
    red, green, blue = rgb
    return green > 200 and abs(red - green) < range_value and abs(blue - green) < range_value

def draw_green_areas(image, green_pixels):
    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)

    for coord in green_pixels:
        draw.point(coord, fill=(0, 255, 0))  # 将绿色像素标记为绿色

    return marked_image

# 打开图片
image_path = "eletrical cabinet/5.jpg"
image = Image.open(image_path)

# 增加范围的值
range_value = 30

# 查找绿色像素的坐标
green_pixels = find_green_pixels(image, range_value)

# 在图片上标记绿色区域并保存
marked_image = draw_green_areas(image, green_pixels)
marked_image.save("marked_image.jpg")

# 显示标记后的图片
marked_image.show()

from PIL import Image
import cv2

def get_pixel_rgb(image_path, x, y):
    # 打开图像
    image = Image.open(image_path)

    # 获取指定位置的像素值
    pixel = image.getpixel((x, y))

    # 检查像素值的模式
    if image.mode == 'RGB':
        # 如果是RGB模式，直接返回像素值
        return pixel
    elif image.mode == 'RGBA':
        # 如果是RGBA模式，忽略Alpha通道
        r, g, b, _ = pixel
        return (r, g, b)
    else:
        # 对于其他模式，将像素值转换为RGB模式后返回
        return pixel[:3]



# 鼠标点击事件的回调函数
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取点击位置的RGB值
        rgb = get_pixel_rgb(image_path, x, y)
        print(f'RGB值：{rgb}')

        # 在图像上显示点击位置
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', image)

# 图像路径
image_path = 'path_to_your_image.jpg'

# 打开图像
image = cv2.imread(image_path)

# 创建窗口并绑定鼠标点击事件回调函数
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

# 显示图像
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 图像路径
image_path = 'facilities/depth_img/5-dpt_beit_large_512.png'

# 鼠标点击位置
click_x = 100
click_y = 200

# 获取指定位置的RGB值
rgb = get_pixel_rgb(image_path, click_x, click_y)
print(f'RGB值：{rgb}')

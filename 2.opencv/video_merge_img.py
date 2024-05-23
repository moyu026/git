import cv2
import numpy as np


video_path = 'file/test.mp4'
cap = cv2.VideoCapture(video_path)

# 读取第一帧作为初始画布
ret, canvas = cap.read()
canvas_height, canvas_width = canvas.shape[:2]

# 定义画布的初始高度和每次添加部分的高度
initial_canvas_height = canvas_height
additional_height = 5

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 5 == 0:
        # 动态调整画布的高度
        canvas_height += additional_height
        new_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # 复制之前画布的内容到新画布
        new_canvas[:initial_canvas_height, :, :] = canvas

        # 在画布下方添加新帧的一小部分
        new_canvas[initial_canvas_height:, :, :] = frame[-additional_height:, :, :]

        # 更新画布
        canvas = new_canvas
        initial_canvas_height = canvas_height

    frame_count += 1

cap.release()


cv2.imwrite('merged_test.jpg', canvas)

# cv2.imshow('Merged Image', canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

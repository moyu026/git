import torch
import torchvision
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

def draw_img(image, label, left_top, right_bottom, color=(255, 0, 0), txt_color=(255, 255, 255)):  # 在图片上画出开关的的位置，以及检测的结果
    draw_img = image.copy()

    pillow_image = Image.fromarray(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pillow_image)
    font = ImageFont.truetype('STZHONGS.TTF', size=40)
    bbox = draw.textbbox((0, 0), label, font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # w, h = draw.textsize(label, font)      # 文字标签的宽和高
    draw.rectangle([left_top[0], left_top[1] - h, left_top[0] + w, left_top[1]], outline=color, fill=color)  # 文字框
    draw.rectangle([left_top[0], left_top[1], right_bottom[0], right_bottom[1]], outline=color, width=2)     # 图片框
    draw.text((left_top[0], left_top[1] - h-5), label, fill=txt_color, font=font)
    result_image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)

    return result_image


ocr = PaddleOCR(use_angle_cls=True, lang="en")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_small_light = torch.hub.load(r'./yolov5/', 'custom', source='local', path='file/yolov5m.pt')
model_small_light.to(device)

cap = cv2.VideoCapture('file/2_682.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = ocr.ocr(frame)
    print(result)
    for i in result[0]:
        left_top = i[0][0]
        right_bottom = i[0][2]
        label = i[1][0]
        frame = draw_img(frame, label, left_top, right_bottom)

    results = model_small_light(frame)
    detections = results.xyxy[0]
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_id = detection.tolist()
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        left_top = (xmin, ymin)
        right_bottom = (xmax, ymax)
        label = results.names[class_id]

        frame = draw_img(frame, label, left_top, right_bottom)

    # 显示当前帧
    cv2.imshow('Video', frame)
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频捕获对象
cap.release()

# 销毁所有窗口
cv2.destroyAllWindows()

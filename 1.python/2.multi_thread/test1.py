import threading
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



def ocr_thread(frame, frame_ocr_result):
    result = ocr.ocr(frame)
    frame_ocr_result['result'] = result
    frame_ocr_result['done'] = True

def detection_thread(frame, frame_detection_result):
    # 假设model_small_light返回的是包含类名信息的结果
    results = model_small_light(frame)
    detections = results.xyxy[0]
    class_names = results.names  # 假设存在这样的属性或方法获取类名列表

    # 将检测结果存储到共享数据结构中，并同时存储类名
    detected_objects = []
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_id = detection.tolist()
        label = class_names[class_id]
        detected_objects.append((label, (xmin, ymin), (xmax, ymax), confidence))

    frame_detection_result['detections'] = detected_objects
    frame_detection_result['done'] = True


if __name__ == '__main__':

    cap = cv2.VideoCapture('file/2_682.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 初始化线程共享数据结构来存储结果
        frame_ocr_result = {'done': False, 'result': None}
        frame_detection_result = {'done': False, 'detections': []}

        # 创建并启动OCR线程
        ocr_thread_obj = threading.Thread(target=ocr_thread, args=(frame, frame_ocr_result))
        ocr_thread_obj.start()

        # 创建并启动YOLOv5检测线程
        detection_thread_obj = threading.Thread(target=detection_thread, args=(frame, frame_detection_result))
        detection_thread_obj.start()

        # 等待两个线程完成
        ocr_thread_obj.join()
        detection_thread_obj.join()

        # 获取并处理每个线程的结果
        for i in frame_ocr_result['result'][0]:
            left_top = i[0][0]
            right_bottom = i[0][2]
            label = i[1][0]
            frame = draw_img(frame, label, left_top, right_bottom)

        for detection in frame_detection_result['detections']:
            label, left_top, right_bottom, confidence = detection
            left_top = tuple(int(coord) for coord in left_top)
            right_bottom = tuple(int(coord) for coord in right_bottom)

            frame = draw_img(frame, label, left_top, right_bottom)

        # 显示当前帧
        cv2.imshow('Video', frame)
        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ...（保留释放资源和关闭窗口部分）
import multiprocessing as mp
import torch.multiprocessing as mp_torch  # 如果需要在GPU上运行YOLOv5模型，需要此导入
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

# 确保在Windows系统下能够共享CUDA上下文（如果在GPU上运行）
# if __name__ == '__main__':
#     mp_torch.set_start_method('spawn')

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

def ocr_process(frame, result_queue):
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    result = ocr.ocr(frame)
    result_queue.put(result)  # 放入结果到队列中

def detection_process(frame, result_queue):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_small_light = torch.hub.load(r'./yolov5/', 'custom', source='local', path='file/yolov5m.pt')
    model_small_light.to(device)
    # 假设model_small_light返回的是包含类名信息的结果
    results = model_small_light(frame)
    detections = results.xyxy[0]
    class_names = results.names  # 假设存在这样的属性或方法获取类名列表

    detected_objects = []
    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_id = detection.tolist()
        label = class_names[class_id]
        detected_objects.append((label, (xmin, ymin), (xmax, ymax), confidence))

    result_queue.put(detected_objects)  # 将检测结果放入队列中

if __name__ == '__main__':


    cap = cv2.VideoCapture(r'F:\PythonWork\0.other\0.study\python\2.multi_thread\file\2_682.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ocr_result_queue = mp.Queue()
        detection_result_queue = mp.Queue()

        ocr_proc = mp.Process(target=ocr_process, args=(frame, ocr_result_queue))
        ocr_proc.start()
        detection_proc = mp.Process(target=detection_process, args=(frame, detection_result_queue))
        detection_proc.start()

        ocr_proc.join()
        detection_proc.join()

        frame_ocr_result = ocr_result_queue.get()
        frame_detection_result = detection_result_queue.get()

        for i in frame_ocr_result[0]:
            left_top = i[0][0]
            right_bottom = i[0][2]
            label = i[1][0]
            frame = draw_img(frame, label, left_top, right_bottom)

        for detection in frame_detection_result:
            label, left_top, right_bottom, confidence = detection
            left_top = tuple(int(coord) for coord in left_top)
            right_bottom = tuple(int(coord) for coord in right_bottom)

            frame = draw_img(frame, label, left_top, right_bottom)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
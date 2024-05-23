import multiprocessing as mp
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np


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
    draw.rectangle([left_top[0], left_top[1], right_bottom[0], right_bottom[1]], outline=color, width=2)  # 图片框
    draw.text((left_top[0], left_top[1] - h - 5), label, fill=txt_color, font=font)
    result_image = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)

    return result_image


def detection_and_queue_process(result_queue):
    cap = cv2.VideoCapture(r'F:\PythonWork\0.other\0.study\python\2.multi_thread\file\2_682.mp4')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load(r'./yolov5/', 'custom', source='local', path='file/yolov5m.pt')
    model.to(device)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0]
        class_names = results.names

        detected_objects = []
        for detection in detections:
            xmin, ymin, xmax, ymax, confidence, class_id = detection.tolist()
            label = class_names[class_id]
            detected_objects.append((label, (xmin, ymin), (xmax, ymax), confidence))

        result_queue.put((frame, detected_objects))

    result_queue.put(None) # 最后一帧为空时，则结束显示程序

    cap.release()


def display_process(result_queue):
    while True:
        frame_data = result_queue.get()
        if frame_data is None:
            break

        frame, detections = frame_data

        for detection in detections:
            frame = draw_img(frame, detection[0], detection[1], detection[2])

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    result_queue = mp.Queue()

    detection_proc = mp.Process(target=detection_and_queue_process, args=(result_queue,))
    display_proc = mp.Process(target=display_process, args=(result_queue,))

    detection_proc.start()
    display_proc.start()

    detection_proc.join()
    display_proc.join()

    cv2.destroyAllWindows()

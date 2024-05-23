import multiprocessing as mp
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


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


def detect(result_queue):
    cap = cv2.VideoCapture(r'F:\PythonWork\0.other\0.study\python\2.multi_thread\file\2_682.mp4')
    model = torch.hub.load(r'./yolov5/', 'custom', source='local', path='file/yolov5m.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = results.xyxy[0]
        class_names = results.names

        detect_object = []

        for detection in detections:
            xmin, ymin, xmax, ymax, confidence, class_id = detection.tolist()
            label = class_names[class_id]
            detect_object.append((label, (xmin, ymin), (xmax, ymax), confidence))

        result_queue.put((frame, detect_object))

    result_queue.put(None)

    cap.release()


def display(result_queue):
    while True:
        frame_data = result_queue.get()
        if frame_data is None:
            break

        frame, detections = frame_data

        for detection in detections:
            frame = draw_img(frame, detection[0], detection[1], detection[2])

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    result_queue = mp.Queue()

    detect_pro = mp.Process(target=detect, args=(result_queue,))
    display_pro = mp.Process(target=display, args=(result_queue,))

    detect_pro.start()
    display_pro.start()

    detect_pro.join()
    display_pro.join()

    cv2.destroyAllWindows()

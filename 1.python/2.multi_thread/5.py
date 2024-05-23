import cv2
from paddleocr import PaddleOCR
import threading
import queue

# 创建一个队列用于存放视频帧
frame_queue = queue.Queue()

ocr = PaddleOCR(use_angle_cls=True, lang="ch")

def read_video(filename, frame_queue):
    cap = cv2.VideoCapture(filename)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 将读取到的帧放入队列中
        frame_queue.put(frame)
    # 当视频读取完毕时，向队列中放入一个特殊值表示结束
    frame_queue.put(None)

def detect_video(frame_queue):
    while True:
        # 从队列中获取一帧，如果没有帧则阻塞等待
        frame = frame_queue.get()
        if frame is None:
            # 遇到特殊值（None）则退出循环
            break
        result = ocr.ocr(frame)
        print(result)

if __name__ == "__main__":
    video_path = r'F:\PythonWork\6.grpc\data/1.mp4'

    t1 = threading.Thread(target=read_video, args=(video_path, frame_queue,))
    t2 = threading.Thread(target=detect_video, args=(frame_queue,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()
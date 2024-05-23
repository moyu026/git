import cv2

def check_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    else:
        print(f"Success: Video file {video_path} opened successfully")
        return True

# 使用示例
video_path = "path/to/your/video.mp4"
check_video(video_path)

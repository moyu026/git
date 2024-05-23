import cv2
from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
import random


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


def get_point(file_path):  # 获取json文件中的开关名称和对应的bbox坐标
    button_dict = {}
    yaban_dict = {}
    samll_light_dict = {}
    poiner_dict = {}
    light_dict = {}
    monitor_dict = {}
    protection_dict = {}
    binglie_dict = {}
    with open(file_path, 'r') as load_f:
        json_dict = json.load(load_f)
        for obj_dict in json_dict['objects']:
            if obj_dict['category'].split('_')[-1] == '压板':
                yaban_dict[obj_dict['category'].split('_')[-2]] = obj_dict['bbox']
            if obj_dict['category'].split('_')[-1] == '旋钮':
                button_dict[obj_dict['category'].split('_')[-2]] = obj_dict['bbox']
            if obj_dict['category'].split('_')[-1] == '小灯':
                samll_light_dict[obj_dict['category'].split('_')[-2]] = obj_dict['bbox']
            # if obj_dict['category'].split('_')[-1] == '指针':
            #     poiner_dict[obj_dict['category'].split('_')[-2]] = obj_dict['bbox']
            if obj_dict['category'].split('_')[-1] == '灯':
                light_dict[obj_dict['category'].split('_')[-2]] = obj_dict['bbox']
            if obj_dict['category'].split('_')[-1] == '高压带电显示器':
                monitor_dict[obj_dict['category'].split('_')[-2]] = obj_dict['bbox']
            if obj_dict['category'].split('_')[-1] == '保护装置':
                protection_dict[obj_dict['category'].split('_')[-2]] = obj_dict['bbox']
            if obj_dict['category'].split('_')[-1] == '并列装置':
                binglie_dict[obj_dict['category'].split('_')[-2]] = obj_dict['bbox']

    return button_dict, yaban_dict, samll_light_dict, poiner_dict, light_dict, monitor_dict, protection_dict, binglie_dict

def judge_detect_name(left_top, right_bottom, small_light_dict):
    # 检测到的灯的方框的中心点洛在标记的一个灯的bbox中，则该检测到的灯的名字就是这个标记的灯的名字
    for small_light_name in small_light_dict:
        small_light_point = small_light_dict[small_light_name]
        xmin, ymin, xmax, ymax = int(small_light_point[0]), int(small_light_point[1]), int(small_light_point[2]), int(small_light_point[3])
        p_x = (right_bottom[0] + left_top[0])/2
        p_y = (right_bottom[1] + left_top[1])/2

        if xmin < p_x < xmax and ymin < p_y < ymax:
            return small_light_name
        else:
            continue

def correct_img(local_img, receive_img): # 接收到的图片与标定图片做匹配并矫正
    # SURF 特征
    surf = cv2.SIFT_create()
    keypoints1, descriptors1 = surf.detectAndCompute(receive_img, None)
    keypoints2, descriptors2 = surf.detectAndCompute(local_img, None)

    # FLANN 匹配
    flann = cv2.FlannBasedMatcher_create()
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 去掉最近邻m和次近邻n的距离比值太大的匹配结果
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    #  计算转换矩阵
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

    # 透视变换
    aligned_image = cv2.warpPerspective(receive_img, H, (local_img.shape[1], local_img.shape[0]))

    return aligned_image


# 比较图片的清晰度
def calculate_image_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness

# 从视频中随机抽取3帧图片
def extract_and_find_sharpest_frame(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_indices = random.sample(range(total_frames), num_frames)

    selected_frames = []
    for frame_index in random_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:
            selected_frames.append(frame)

    cap.release()

    max_sharpness = float('-inf')
    sharpest_frame = None

    for frame in selected_frames:
        sharpness = calculate_image_sharpness(frame)
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            sharpest_frame = frame

    return sharpest_frame
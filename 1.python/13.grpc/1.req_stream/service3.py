import time
import grpc
import detect_pb2_grpc as pb2_grpc
from concurrent import futures
import cv2
import numpy as np

import urllib.request
from util.plot_img import *
from detect.detect import Detection, Classsification, Pointer


class Detect(pb2_grpc.Electrical_Cabinet_DetectServicer):
    def __init__(self):
        self.label_dir = './data/label'
        self.local_img_dir = './data/image'
        self.model_button_path = './weights/resNet18_button2.pth'
        self.model_yaban_path = './weights/resNet50_yaban1.pth'
        self.model_small_light_path = './weights/detect_small_light.pt'
        # self.model_light_path = './weights/detect_light.pt'
        self.model_pointer_path = './weights/pointer_vgg_80.pth'

    def detection(self, request_iterator, context):
        for request in request_iterator:
            img_path = request.path
            img_name = request.name
            # print(img_name)

            local_img_path = self.local_img_dir + "\\" + img_name + '.jpg'
            local_img = cv2.imdecode(np.fromfile(local_img_path, dtype=np.uint8), -1)

            label_path = self.label_dir + '\\' + img_name.split('.')[0] + '.json'

            if img_path.split('.')[-1] == 'mp4':
                img = extract_and_find_sharpest_frame(img_path)
            else:
                resp = urllib.request.urlopen(img_path)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                img = cv2.imdecode(image, cv2.IMREAD_COLOR)

            total_result = {}

            # 图片位置矫正
            img = correct_img(local_img, img)

            # 读取开关的标签信息：{开关名称：坐标}
            button_dict, yaban_dict, small_light_dict, pointer_dict, light_dict, monitor_dict, protection_dict, binglie_dict = get_point(label_path)

            # 分类模型
            clsss_mathod = Classsification(self.model_button_path, self.model_yaban_path)
            button_result, img = clsss_mathod.button_classify(img, button_dict)
            yaban_result, img = clsss_mathod.yaban_classify(img, yaban_dict)
            if len(button_result['旋钮']) != 0:
                total_result.update(button_result)
            if len(yaban_result['压板']) != 0:
                total_result.update(yaban_result)

            # 小灯检测模型
            detect_method = Detection(self.model_small_light_path)
            small_light_result, img = detect_method.small_light_detect(img, small_light_dict, monitor_dict, protection_dict, binglie_dict)
            if len(small_light_result) != 0:
                total_result.update(small_light_result)

            # 指针模型
            if len(pointer_dict) != 0:
                pointer_method = Pointer(self.model_pointer_path)
                pointer_result, img = pointer_method.pointer_detect(img, pointer_dict)

                if len(pointer_result) != 0:
                    total_result.update(pointer_result)

            # result_json_name = 'result' + '\\' + str((img_path.split('/')[-1]).split('.')[0]) + '.json'
            result_json_name = 'result' + '\\' + img_name + '.json'
            with open(result_json_name, 'w', encoding='utf-8') as json_file:
                json.dump(total_result, json_file, ensure_ascii=False, indent=4)

            # save_img_name = 'result' + '\\' + str((img_path.split('/')[-1]).split('.')[0]) + '.jpg'
            save_img_name = 'result' + '\\' + img_name + '.jpg'
            cv2.imwrite(save_img_name, img)

            print(total_result)


def run():
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4)
    )

    pb2_grpc.add_Electrical_Cabinet_DetectServicer_to_server(Detect(), grpc_server)
    grpc_server.add_insecure_port('127.0.0.1:5000')
    print('server start...')
    grpc_server.start()

    try:
        while 1:
            time.sleep(3600)
    except KeyboardInterrupt:
        grpc_server.stop(0)


if __name__ == '__main__':
    run()

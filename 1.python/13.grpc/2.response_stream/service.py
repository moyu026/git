import time
import grpc
# import detect_pb2_grpc as pb2_grpc
import arithmetic_pb2 as pb2
import arithmetic_pb2_grpc as pb2_grpc
from concurrent import futures
import cv2
import numpy as np

import urllib.request
from util.plot_img import *
# from detect.detect import Detection, Classsification, Pointer
from detect.detect import Detection, Classsification
import os
import requests

root_directory = os.getcwd()
url = 'http://192.168.101.152:9010/file/upload'

class Detect(pb2_grpc.ArithmeticServiceServicer):
    def __init__(self):
        self.label_dir = 'data/label1'
        self.local_img_dir = 'data/image'
        self.model_button_path = './weights/resNet18_button2.pth'
        self.model_yaban_path = './weights/resNet50_yaban1.pth'
        self.model_small_light_path = './weights/detect_small_light.pt'
        # self.model_light_path = './weights/detect_light.pt'
        self.model_pointer_path = './weights/pointer_vgg_80.pth'


    def analysis(self, request, context):
        for task_asset in request.task_asset:
            data_id = task_asset.id
            data_type = task_asset.type
            img_path = 'http://192.168.101.152:9010/file' + '/' + task_asset.location
            img_name = task_asset.device_name

            local_img_path = os.path.join(root_directory, self.local_img_dir, img_name + '.jpg')

            if not os.path.exists(local_img_path):  # 先判断本地是否有标定过这个名字的图片
                det_location = None
                ori_location = None
                total_result = {}
            else:
                local_img = cv2.imdecode(np.fromfile(local_img_path, dtype=np.uint8), -1)
                label_path = os.path.join(root_directory, self.label_dir, img_name.split('.')[0] + '.json')

                if data_type == 'VIDEO':
                    img = extract_and_find_sharpest_frame(img_path)
                    ori_img = img.copy()
                else:
                    resp = urllib.request.urlopen(img_path)
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    ori_img = img.copy()

                # 图片位置矫正
                img = correct_img(local_img, img)

                # 读取开关的标签信息：{开关名称：坐标}
                button_dict, yaban_dict, small_light_dict, pointer_dict, light_dict, monitor_dict, protection_dict, binglie_dict = get_point(label_path)

                # 分类模型
                clsss_mathod = Classsification(self.model_button_path, self.model_yaban_path)
                button_result, img = clsss_mathod.button_classify(img, button_dict)
                yaban_result, img = clsss_mathod.yaban_classify(img, yaban_dict)

                # 小灯检测模型
                detect_method = Detection(self.model_small_light_path)
                small_light_result, img = detect_method.small_light_detect(img, small_light_dict, monitor_dict, protection_dict, binglie_dict)

                # # 指针模型
                # if len(pointer_dict) != 0:
                #     pointer_method = Pointer(self.model_pointer_path)
                #     pointer_result, img = pointer_method.pointer_detect(img, pointer_dict)
                #
                #     total_result = {**button_result, **yaban_result, **small_light_result, **pointer_result}
                #
                # else:
                    # 总的检测结果
                total_result = {**button_result, **yaban_result, **small_light_result}

                # result_json_name = os.path.join(root_directory, 'results', 'result_json', img_name + '.json')
                # with open(result_json_name, 'w', encoding='utf-8') as json_file:
                #     json.dump(total_result, json_file, ensure_ascii=False, indent=2)

                save_img_name = os.path.join(root_directory, 'results', 'result_img', img_name + '.jpg')
                ori_img_name = os.path.join(root_directory, 'results', 'ori_img', img_name + '.jpg')
                cv2.imwrite(save_img_name, img)
                cv2.imwrite(ori_img_name, ori_img)

                print(data_id, total_result)

                # 上传分析的图片和原图片
                ori_files = {'file': open(ori_img_name, 'rb')}
                ori_response = requests.post(url, files=ori_files)
                ori_response_json = ori_response.json()  # 解析JSON格式的响应内容
                ori_location = str(ori_response_json.get("location"))

                det_files = {'file': open(save_img_name, 'rb')}
                det_response = requests.post(url, files=det_files)
                det_response_json = det_response.json()
                det_location = str(det_response_json.get("location"))


            response = pb2.TaskAssetResponse(
                id=task_asset.id,
                location_wrap=[
                        pb2.LocationWrap(
                        location=det_location,
                        type=pb2.AnalysisType.ANALYSIS
                    ), pb2.LocationWrap(
                        location=ori_location,
                        type=pb2.AnalysisType.ORIGINAL
                    )
                ],
                attributes=total_result
            )
            yield response


def run():
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4)
    )

    pb2_grpc.add_ArithmeticServiceServicer_to_server(Detect(), grpc_server)
    grpc_server.add_insecure_port('0.0.0.0:9190')
    print('server start...')
    grpc_server.start()

    try:
        while 1:
            time.sleep(3600)
    except KeyboardInterrupt:
        grpc_server.stop(0)


if __name__ == '__main__':
    run()

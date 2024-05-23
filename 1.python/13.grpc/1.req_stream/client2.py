import time

import grpc
import detect_pb2_grpc
import detect_pb2


def test():
    while 1:
        time.sleep(1)
        img_path = [
                    'data/video/1P2Ⅱ段压变避雷器开关柜.mp4',
                    'data/video/116金陆线开关柜.mp4',
                    'data/video/151#1接地变开关柜.mp4',
                    'data/video/162#2电容器开关柜.mp4'] # https://free.wzznft.com/i/2024/01/02/h9qfoo.jpg
        # name_list = ['1P2Ⅱ段压变避雷器开关柜', '116金陆线开关柜', '151#1接地变开关柜', '162#2电容器开关柜', '111', '162#2电容器开关柜']
        name_list = ['1P2Ⅱ段压变避雷器开关柜', '116金陆线开关柜', '151#1接地变开关柜', '162#2电容器开关柜']
        for url, name in zip(img_path, name_list):
            yield detect_pb2.DetectionReq(path=url, name=name)


channel = grpc.insecure_channel('127.0.0.1:5000')
client = detect_pb2_grpc.Electrical_Cabinet_DetectStub(channel)

response = client.detection(test())

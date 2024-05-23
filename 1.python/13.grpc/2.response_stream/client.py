import time

import grpc
# import detect_pb2_grpc
# import detect_pb2
import arithmetic_pb2_grpc
import arithmetic_pb2

channel = grpc.insecure_channel('192.168.102.68:9190')
client = arithmetic_pb2_grpc.ArithmeticServiceStub(channel)


task_assets = [
    arithmetic_pb2.TaskAsset(
        id=1,
        location='anonymous/097e72059bf6437ba9bcb565e2551bed/1P2%E2%85%A1%E6%AE%B5%E5%8E%8B%E5%8F%98%E9%81%BF%E9%9B%B7%E5%99%A8%E5%BC%80%E5%85%B3%E6%9F%9C.mp4',
        device_name='1P2Ⅱ段压变避雷器开关柜',
        type="VIDEO"
    ),
    arithmetic_pb2.TaskAsset(
        id=2,
        location='data/video/116金陆线开关柜.mp4',
        device_name='116金陆线开关',
        type="VIDEO"
    )
]

request = arithmetic_pb2.TaskAssets(task_asset=task_assets)

responses = client.analysis(request)

for response in responses:
        print("Received TaskAssetResponse:")
        print(f"ID: {response.id}")
        for location_wrap in response.location_wrap:
            print(f"Location: {location_wrap.location}, Type: {location_wrap.type}")
        print(f"Attributes: {response.attributes}")












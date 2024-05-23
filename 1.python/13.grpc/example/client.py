# client.py
import random
import time

import grpc
import hello_pb2_grpc as pb2_grpc
import hello_pb2 as pb2

def test():
    index = 0
    while 1:
        time.sleep(1)
        data = str(random.random())
        if index == 5:  # 客户端主动断开与服务器端的连接
            break
        print(index)
        index += 1
        yield pb2.TestClientSendStreamRequest(data=data)
def run():
    conn = grpc.insecure_channel('127.0.0.1:5000')  # 添加ip和端口
    client = pb2_grpc.BililiStub(channel=conn)     # 添加客户端BililiStub

    # response = client.Hello(pb2.classifyReq(          # 客户端调用Hello函数，使用pb2.HelloReq传递参数
    #         name='hello fly',
    #         age=33
    #     ),)
    # print(response.result)    # 打印结果：response.result

    response = client.TestClientRecStream(pb2.TestClientRecStreamRequest(
        data='close'
    ))
    for item in response:
        print(item.result)

    # response = client.TestClientSendStream(test())
    # print(response.result)

    # response = client.TestTwoWayStream(test())
    # for res in response:
    #     print(res.result)

if __name__ == '__main__':
    run()
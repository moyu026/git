# service.py
import time
import grpc
import hello_pb2 as pb2
import hello_pb2_grpc as pb2_grpc
from concurrent import futures


class Bili(pb2_grpc.BililiServicer):   # 定义服务, 类名任意，会继承BililiServicer这个服务
    def Hello(self, request, context):  # 服务中的函数， 函数名就是服务中的函数名
        name = request.name    # 用request取函数中的参数
        age = request.age

        result = f'my name is {name}, i am {age} year old'
        return pb2.HelloReply(result=result)        # 返回服务

    def TestClientRecStream(self, request, context):
        index = 0
        while context.is_active: # 判断客户端是否活跃，如果客户端活跃才响应连接
            data = request.data

            if data == 'close':
                print('data is close, request will cancel')
                context.cancel()
            time.sleep(1)
            index += 1
            result = 'send %d %s' % (index, data)
            print(result)
            yield pb2.TestClientRecStreamResponse(
                result = result
            )
    def TestClientSendStream(self, request_iterator, context):
        index = 0
        for request in request_iterator:
            print(request.data, ';', index)
            if index == 10:  # 服务器断开与客户端的连接
                break
            index += 1

        return pb2.TestClientSendStreamResponse(result='ok')

    def TestTwoWayStream(self, request_iterator, context):
        index = 0
        for request in request_iterator:
            data = request.data
            if index == 3:
                context.cancel()
            index += 1
            yield pb2.TestTwoWayStreamResponse(result='service send client %s' % data)

def run():    # 启动服务
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),  # 服务中的线程
    )
    pb2_grpc.add_BililiServicer_to_server(Bili(), grpc_server) # 把Bili()这个类注册到服务中， 添加的是上面定义的类的名称
    grpc_server.add_insecure_port('0.0.0.0:5000')              # 添加ip和端口号
    print('server start..')
    grpc_server.start()    # 启动服务

    try:                  # 保持启动状态
        while 1:
            time.sleep(3600)
    except KeyboardInterrupt:
        grpc_server.stop(0)


if __name__ == '__main__':
    run()
import requests

# 指定的服务器地址和端口以及上传的文件路径
url = 'http://192.168.101.152:9010/file/upload'
image_path = 'results/result_img/1P2Ⅱ段压变避雷器开关柜.jpg'

# 准备要上传的文件
files = {'file': open(image_path, 'rb')}

# 发送 HTTP POST 请求
response = requests.post(url, files=files)


# 检查响应
if response.status_code == 200:
    print("上传成功！")
    response_json = response.json()  # 解析JSON格式的响应内容

    # 获取"location"字段的值
    location = response_json.get("location")
    if location:
        print(f"获取到的location信息为：{location}")

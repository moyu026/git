# 导入 requests 包
import requests

# 发送请求
x = requests.request('get', 'https://www.bing.com/images/search?q=%e5%a3%81%e7%ba%b8&form=HDRSC2&first=1')

# 返回网页内容
print(x.status_code)
print(x.text)
# -*- coding:utf-8 -*-
# coding=gbk
import os
import sys
import time
import urllib
import requests
import re
from bs4 import BeautifulSoup
import time

header = {
    'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 UBrowser/6.1.2107.204 Safari/537.36'
}
url = "https://cn.bing.com/images/async?q={0}&first={1}&count={2}&scenario=ImageBasicHover&datsrc=N_I&layout=ColumnBased&mmasync=1&dgState=c*9_y*2226s2180s2072s2043s2292s2295s2079s2203s2094_i*71_w*198&IG=0D6AD6CBAF43430EA716510A4754C951&SFX={3}&iid=images.5599"


# url = 'https://cn.bing.com/images/search?q=yard&qs=n&form=QBILPG&sp=-1&lq=0&pq=yar&sc=10-3&cvid=15F8CE5420194DA1AE2290DBD5E97410&ghsh=0&ghacc=0&first=1'

def getImage(url, count):
    try:
        time.sleep(0.5)
        urllib.request.urlretrieve(url, path + str(count + 1) + '.jpg')
    except Exception as e:
        time.sleep(1)
        print("error,skip")
    else:
        print("success save " + str(count + 1) + "img")


def findImgUrlFromHtml(html, rule, url, key, first, loadNum, sfx, count):
    soup = BeautifulSoup(html, "lxml")
    link_list = soup.find_all("a", class_="iusc")
    url = []
    for link in link_list:
        result = re.search(rule, str(link))

        url = result.group(0)

        url = url[8:len(url)]

        getImage(url, count)
        count += 1

    return count


def getStartHtml(url, key, first, loadNum, sfx):
    page = urllib.request.Request(url.format(key, first, loadNum, sfx), headers=header)
    html = urllib.request.urlopen(page)
    return html


if __name__ == '__main__':
    name = '壁纸'  # 图片关键词
    path = 'images/'  # 图片保存路径
    countNum = 1  # 爬取数量
    key = urllib.parse.quote(name)
    first = 1  # 表示从第几张图片开始爬取
    loadNum = 1  # 每次加载多少张图片
    sfx = 1
    count = 0
    rule = re.compile(r"\"murl\"\:\"http\S[^\"]+")
    if not os.path.exists(path):
        os.makedirs(path)
    while count < countNum:
        html = getStartHtml(url, key, first, loadNum, sfx)
        count = findImgUrlFromHtml(html, rule, url, key, first, loadNum, sfx, count)
        first = count + 1
        sfx += 1

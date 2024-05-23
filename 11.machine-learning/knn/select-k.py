### 用交叉验证的方法确定K值
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

imagePaths = getListOfFiles("./dataset/") ## Folder structure: datasets --> sub-folders with labels name
#print(imagePaths)

data = []
labels = []
c = 0 ## to see the progress
for image in imagePaths:

    label = os.path.split(os.path.split(image)[0])[1]
    labels.append(label)

    img = cv2.imread(image)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    data.append(img)
    c = c+1
    print(c)

#print(labels)

# encode the labels as integer
x = np.array(data)

x_size = x.shape[0]
x = x.reshape(x_size, -1)  # knn算法处理的是二维数据，对图片数据降维

labels = np.array(labels)

le = LabelEncoder()
y = le.fit_transform(labels)

k_range = range(1, 10)
k_error = []
#循环，取k=1到k=31，查看误差效果
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)  # n_neighbors:查询使用的邻居数。就是k-NN的k的值，选取最近的k个点。
    #cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
    scores = cross_val_score(knn, x, y, cv=6, scoring='accuracy')  # knn:需要使用交叉验证的算法, x:输入样本数据， y:样本标签, cv:交叉验证折数或可迭代的次数,将数据分成6个子集，然后进行6次交叉验证，每次使用不同的子集作为测试集。 scoring： 交叉验证的验证方式,不同的评价方法，会产生不同的评价结果。
    k_error.append(1 - scores.mean())

#画图，x轴为k值，y值为误差值
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()
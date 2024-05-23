### 用交叉验证的方法确定K值

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#读取鸢尾花数据集
iris = load_iris()
x = iris.data
y = iris.target
k_range = range(1, 31)
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
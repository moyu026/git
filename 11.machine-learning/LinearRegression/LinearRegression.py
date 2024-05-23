import numpy as np


class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # 在特征矩阵中添加一列全为1的列，对应截距项
        X_augmented = np.hstack((np.ones((X.shape[0], 1)), X))

        # 使用最小二乘法计算权重
        self.weights = np.linalg.inv(X_augmented.T.dot(X_augmented)).dot(X_augmented.T).dot(y)

        # 截距项即为权重的第一个元素
        self.bias = self.weights[0]

        # 去除权重中的截距项
        self.weights = self.weights[1:]

    def predict(self, X):
        return X.dot(self.weights) + self.bias


# 示例数据
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 3, 4, 5])

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 预测新数据
X_test = np.array([[5], [6]])
predictions = model.predict(X_test)
print(predictions)

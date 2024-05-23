import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=100000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # 初始化权重和截距项
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 梯度下降优化权重
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # 计算梯度
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # 更新权重和截距项
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        print(y_pred)
        return (y_pred > 0.5).astype(int)

# 示例数据，包括两个特征和二分类标签
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

# 创建并训练逻辑回归模型
model = LogisticRegression()
model.fit(X, y)

# 预测新数据
X_test = np.array([[5, 6], [1, 1]])
predictions = model.predict(X_test)
print(predictions)

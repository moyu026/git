import numpy as np

class SupportVectorMachine:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def _hinge_loss(self, y, y_pred):
        return np.maximum(0, 1 - y * y_pred)

    def _svm_loss(self, y, y_pred):
        return np.mean(self._hinge_loss(y, y_pred)) + self.lambda_param * np.sum(self.weights ** 2)

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # 初始化权重和截距项
        self.weights = np.zeros(num_features)
        self.bias = 0

        # 梯度下降优化权重
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = np.sign(linear_model)

            # 计算梯度
            dw = (1 / num_samples) * (np.dot(X.T, (y_pred * y < 1).astype(int) * -y) + 2 * self.lambda_param * self.weights)
            db = (1 / num_samples) * np.sum((y_pred * y < 1).astype(int) * -y)

            # 更新权重和截距项
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return np.sign(linear_model)

# 示例数据，包括两个特征和二分类标签
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([-1, -1, 1, 1])

# 创建并训练支持向量机模型
model = SupportVectorMachine()
model.fit(X, y)

# 预测新数据
X_test = np.array([[5, 6], [1, 1]])
predictions = model.predict(X_test)
print(predictions)

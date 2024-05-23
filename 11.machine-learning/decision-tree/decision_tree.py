import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _entropy(self, y):
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, y, y_left, y_right):
        p = len(y_left) / len(y)
        return self._entropy(y) - p * self._entropy(y_left) - (1 - p) * self._entropy(y_right)

    def _split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _best_split(self, X, y):
        best_feature, best_threshold, best_info_gain = None, None, -1
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                info_gain = self._information_gain(y, y_left, y_right)
                if info_gain > best_info_gain:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_info_gain = info_gain
        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples == 0 or num_features == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return np.argmax(np.bincount(y))
        if len(np.unique(y)) == 1:
            return y[0]

        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.argmax(np.bincount(y))

        X_left, X_right, y_left, y_right = self._split(X, y, best_feature, best_threshold)
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            feature_index, threshold, left_subtree, right_subtree = node
            if inputs[feature_index] <= threshold:
                node = left_subtree
            else:
                node = right_subtree
        return node

# 示例数据
# X = np.array([
#     [2, 3],
#     [1, 1],
#     [4, 5],
#     [6, 7],
#     [3, 3],
#     [7, 8]
# ])
# y = np.array([0, 0, 1, 1, 0, 1])
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

# 创建决策树并训练
tree = DecisionTree(max_depth=3)
tree.fit(X, y)

# 预测新数据
# X_test = np.array([
#     [5, 5],
#     [1, 2]
# ])
X_test = np.array([[5, 6], [1, 1]])
predictions = tree.predict(X_test)
print(predictions)

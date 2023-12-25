import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# 自定义SVM类
class SVM:
    def __init__(self, sigma=1, C=1, kind="linear"):
        assert kind in ["linear", "gaussian"]
        self.sigma = sigma
        self.C = C
        gaussian = lambda x, z: np.exp(-0.5 * np.sum((x - z)**2) / (self.sigma**2))
        linear = lambda x, z: np.sum(x * z)
        self.kernel = linear if kind == "linear" else gaussian
    
    def fit(self, X, y):
        mat = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i, X.shape[0]):
                result = self.kernel(X[i], X[j])
                mat[i, j] = result
                mat[j, i] = result
        P = mat * (y.reshape(-1, 1) @ y.reshape(1, -1))
        q = -1 * np.ones(X.shape[0]).reshape(-1, 1)
        
        G = np.vstack([-np.identity(X.shape[0]), np.identity(X.shape[0])])
        h = np.hstack([np.zeros(X.shape[0]), np.ones(X.shape[0]) * self.C]).reshape(-1, 1)
        
        A = y.reshape(1, -1)
        b = np.zeros(1).reshape(-1, 1)
        
        [P, q, G, h, A, b] = [matrix(i, i.shape, "d") for i in [P, q, G, h, A, b]]
        result = solvers.qp(P, q, G, h, A, b)
        self.A = np.array(result["x"])
        support_vector_index = np.where(self.A > 1e-4)[0]
        self.support_vectors = X[support_vector_index]
        self.support_vector_as = self.A[support_vector_index, 0]
        self.support_vector_ys = y[support_vector_index]
        for i, a in enumerate(self.A):
            if a > 0 + 1e-4 and a < self.C - 1e-4:
                self.b = y[i] - np.sum(self.A.ravel() * y * mat[i])
                break
    
    def predict(self, X):
        preds = []
        for x in tqdm(X):
            Ks = [self.kernel(x, support_vector) for support_vector in self.support_vectors]
            pred = np.sum(self.support_vector_as * self.support_vector_ys * Ks) + self.b
            pred = 1 if pred >= 0 else -1
            preds.append(pred)
        return np.array(preds)

    def score(self, X, y):
        return np.sum(self.predict(X) == y) / len(y)

# 加载数据集
file_path = r'机器学习/实验课/data/UCI-Heart-Disease.csv'
data = pd.read_csv(file_path)

# 准备SVM的数据
X = data.drop('target', axis=1)  # 特征
y = data['target']  # 目标变量
y = y.replace(0, -1)  # 将“0”替换为“-1”，以便与支持向量机兼容。

# 将数据分为训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义两种不同的核函数进行实验：线性核和高斯核（RBF）
kernel_types = ["linear", "gaussian"]

# 存储每种核函数的准确率
accuracies = {}

# 对每种核函数进行训练和测试
for kernel in kernel_types:
    # 初始化SVM模型
    svc = SVM(kind=kernel)
    # 训练模型
    svc.fit(X_train.values, y_train.values)
    # 计算并存储准确率
    accuracies[kernel] = svc.score(X_test.values, y_test.values)

# 输出每种核函数的准确率
for kernel, acc in accuracies.items():
    print(f"核函数 {kernel}: 准确率 {acc:.2f}")
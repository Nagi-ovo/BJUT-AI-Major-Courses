# %%
import numpy as np
from sklearn.datasets import fetch_openml
from scipy.ndimage.interpolation import shift
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# %%
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.uint8)
X = np.array(X)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# %%
print(X_test.shape)

# %%
# 数据增强函数：对角、水平、垂直位移
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

# %%
print("Creating Augmented Dataset...")

# 初始化增强数据集
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

# 应用水平和垂直位移
for dx, dy in tqdm(((1, 0), (-1, 0), (0, 1), (0, -1)), desc="Applying horizontal and vertical shifts"):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

# 应用对角线位移
for dx, dy in tqdm([(1, 1), (-1, -1), (1, -1), (-1, 1)], desc="Applying diagonal shifts"):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

# 打乱增强数据集
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = np.array(X_train_augmented)[shuffle_idx]
y_train_augmented = np.array(y_train_augmented)[shuffle_idx]

print("Creating Augmented Dataset completed")

# %%
# 训练 XGBoost 模型
print("Training XGBoost model...")
xgb_clf = xgb.XGBClassifier(objective='multi:softprob', num_class=10, n_estimators=500)
xgb_clf.fit(X_train_augmented, y_train_augmented, 
            eval_set=[(X_test, y_test)], 
            eval_metric="mlogloss", 
            early_stopping_rounds=10,  # 如果在10轮迭代内验证集上的性能没有改善，则停止训练
            verbose=True)

# 预测和评估
y_pred = xgb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

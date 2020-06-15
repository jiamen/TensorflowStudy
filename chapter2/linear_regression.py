
import tensorflow as tf
import numpy as np

data = []   # 保存样本集的列表
for i in range(100):    # 循环采样100个点
    x = np.random.uniform(-10., 10.)    # 随机采样输入x
    eps = np.random.normal(0., 0.1)     # 采样高斯噪声
    y = 1.477 * x + 0.089 + eps         # 得到模型的输出
    data.append([x, y])

data = np.array(data)       # 得到





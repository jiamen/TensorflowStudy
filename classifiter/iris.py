
import numpy as np                      # 矩阵运算
import pandas as pd                     # 数据分析，从csv, excel中读取数据，做筛选和排序之类的工作
from keras.models import Sequential             # 序列，串行类
from keras.layers import Dense	                # 隐含层的节点与之前之后的节点都有高密度的连接
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold	    # K-1个训练数据集，1个测试数据集，重复K次
from sklearn.preprocessing import LabelEncoder  # 将字符串标签转换为数字
from keras.models import model_from_json        # 将训练好的模型存储或者读取出来


# reproducibility
seed = 13
np.random.seed(seed)

# load data
df = pd.read_csv('datasets_19_420_Iris.csv')
X = df.values[:, 1:5].astype(float)             # 4维数据：花萼花瓣的长和宽
Y = df.values[:, 5]

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)


# define a network
def baseline_model():
    model = Sequential()        # model 是 Sequential类的对象。 Sequential 是⼀个有序的容器，⽹络层将按照在传⼊ Sequential 的顺序依次被添加到计算图中。
    model.add(Dense(7, input_dim=4, activation='tanh'))         # 输入层到隐藏层
    model.add(Dense(3, activation='softmax'))                   # 输出层维度为类别维数=3
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])   # sgd随机梯度下降优化器
                # 衡量网络的输出与真正类别的差值
    # 用 model.compile 激励神经网络，选择损失函数、求解方法、度量方法
    # 优化器optimizer，可以是默认的，也可以是我们在上一步定义的。 损失函数，分类和回归问题的不一样，用的是交叉熵。 metrics，里面可以放入需要计算的 cost，accuracy，score 等。
    return model


estimator =  KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=1, verbose=1)


# evalute
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)        # 把150个样本数据分为10份，每次取1份作为测试数据集, 看模型预测的结果
result = cross_val_score(estimator, X, Y_onehot, cv=kfold)
print("\nAccuray of cross validation, mean %.2f, std %.2f" % (result.mean(), result.std()))   # 均值和方差, 对10次模型交叉验证的分类结果取均值和方差


# save model
# 训练模型
# 用到的是 fit 函数，把训练集的 x 和 y 传入之后，nb_epoch 表示把整个数据训练多少次，batch_size 每批处理32个。
estimator.fit(X, Y_onehot)
model_json = estimator.model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

estimator.model.save_weights("model.h5")
print("saved model to disk")


# load model and use it for prediction
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)   # 加载已经训练好的模型，但只是加载了网络结构, 还没有加载权值
loaded_model.load_weights("model.h5")
print("loaded model from disk")


predicted = loaded_model.predict(X)
print("predicted probality: " + str(predicted))

predicted_label = loaded_model.predict_classes(X)
print("\npredicted label: " + str(predicted_label))



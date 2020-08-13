import pickle                       # python专门把对象数据存储成二进制文件的工具
import numpy as np                  # 做线性代数运算和数值运算
import matplotlib.pyplot as plt     # 画图

np.random.seed(23)                  # 为了保证可重复性，比如为了产生随机数， 不选择的话，每次程序运行的结果不同，所以为了实验结果的可重复性，这里设置了数值

''' 1. 加载待训练的交通标识数据 '''
def load_traffic_sign_data(training_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    return X_train, y_train


''' 2. 显示读取到的X_train数据中，每种交通标识中随机找一个例子显示出来看一下 '''
def show_random_samples(X_train, y_train, n_classes):   # n_classes表示在X_train中有多少种不同的交通标识
    # show a random sample from each class of the traffic sign dataset
    rows, cols = 4, 12
    fig, ax_array = plt.subplots(rows, cols)
    plt.suptitle('Random Samples (one per class)')
    for class_idx, ax in enumerate(ax_array.ravel()):   # 列举，枚举
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        if class_idx < n_classes:
            # show a random image of the current class
            cur_X = X_train[y_train == class_idx]
            cur_img = cur_X[np.random.randint(len(cur_X))]
            ax.imshow(cur_img)
            ax.set_title('{:02d}'.format(class_idx+1))  # 小标题从01开始显示
        else:
            ax.axis('off')

    # hide both x and y ticks
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.draw()
    # plt.show()


''' 3. 显示不同类别中各自数据的数量，用柱状图显示 '''
def show_classes_distribution(n_classes, y_train, n_train, ntest_classes, y_test, n_test):
    # bar-chart of classes distribution
    train_distribution = np.zeros(n_classes)
    test_distribution  = np.zeros(ntest_classes)
    for c in range(n_classes):  # 各类数据的百分比
        train_distribution[c] = np.sum(y_train == c) / n_train  # 各类数量 / 训练总数量
    for c in range(ntest_classes):  # 各类数据的百分比
        test_distribution[c] = np.sum(y_test == c) / n_test     # 各类数量 / 训练总数量

    fig, ax = plt.subplots()
    ind = np.arange(len(train_distribution)+len(test_distribution))
    col_width = 1
    # bar_train1  = ax.bar(np.arange(n_classes)+np.arange(ntest_classes), train_distribution, width=col_width, color='r', edgecolor = 'white', label='red', lw=1)
    # bar_train2 = ax.bar(np.arange(ntest_classes)+np.arange(ntest_classes)+col_width, test_distribution, width=col_width, color='b', edgecolor = 'white', label='blue', lw=1)
    bar_train1 = ax.bar(np.arange(n_classes), train_distribution, width=col_width/2, color='r', label='red')
    bar_train2 = ax.bar(np.arange(n_classes)+col_width/2, test_distribution, width=col_width/2, color='b', label='blue')
                        # 显示位置                              要显示的数组      柱状图每个柱体宽度      颜色

    ax.set_ylabel('Percentage')                                                 # 设置纵轴坐标名称
    ax.set_xlabel('Class Label')                                                # 设置横轴坐标名称
    ax.set_title('Distribution')
    ax.set_xticks(np.arange(0, n_classes, 5) + col_width/2)                     # 设置x轴坐标显示 位置
    ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])    # 设置x轴坐标范围 值
    plt.legend(["train", "test"])                                               # 设置图例
    plt.show()



if __name__ == "__main__":
    X_train, y_train = load_traffic_sign_data('./traffic-signs-data/train.p')
    X_test, y_test = load_traffic_sign_data('./traffic-signs-data/test.p')

    # Number of examples
    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape
    imgtest_shape = X_test[0].shape

    # How many classes?
    n_classes = np.unique(y_train).shape[0]
    ntest_classes = np.unique(y_test).shape[0]


    print("训练数据集的数据个数 =", n_train)
    print("图像尺寸  =", image_shape)
    print("类别数量 =", n_classes)

    print("测试数据集的数据个数 =", n_test)
    print("图像尺寸  =", imgtest_shape)
    print("类别数量 =", ntest_classes)


    show_random_samples(X_train, y_train, n_classes)
    show_classes_distribution(n_classes, y_train, n_train, ntest_classes, y_test, n_test)
    #show_random_samples(X_test, y_test, ntest_classes)
    #show_classes_distribution(ntest_classes, y_test, n_test)




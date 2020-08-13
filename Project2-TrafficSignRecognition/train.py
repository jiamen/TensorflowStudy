from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
import cv2
from sklearn.model_selection import train_test_split        # sklearn的模型选择，把某个数据集分为测试数据和训练数据
from keras.preprocessing.image import ImageDataGenerator    #
from keras.callbacks import ModelCheckpoint
from keras import utils
from keras.models import load_model
from util import *

np.random.seed(23)      # 为了可以复现结果


def get_mean_std_img(X):
    # convert from RGB tp YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])

    # adjust image contrast
    X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X])

    X = np.float32(X)

    mean_img = np.mean(X, axis=0)       # 我们希望突出不同交通标识图像的差异，因此希望将这种差异传递进神经网络进行训练
    std_img  = (np.std(X, axis=0) + np.finfo('float32').eps)    # 防止方差为0，又出现在分母上，因此这里加上一个小数

    return mean_img, std_img


''' 1. 图像预处理 '''
def preprocess_features(X, mean_img, std_img):
    # X是图像列表或者numpy array，能否只考虑图片中不同地方的亮度判断不同类别，不需要判断颜色，一是减少数据量；二是把不想要的色彩信息去掉（因为色彩一般是蓝底白字，没有太多信息，神经网络集中处理形状信息）
    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])
            # expand_dims把一维通道扩展为2维             # 只保留Y这个亮度通道，忽略色彩信息         X数据集中的每张图像
    # 外围的[] 表示list， np.array 把 [] list 转换为 array


    # 直方图均衡化  (Frequency(重复率，频率) Histogram)
    X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X])

    X = np.float32(X)

    # standardize features， 标准化处理，使网络收敛更快
    X -= mean_img
    X /= std_img
    # 输出神经的每张图像中的像素都可以认为是一个特征，比如有些图像上固定位置的像素总是黑的或者是白的，也就是有的地方变化小，有的地方变化大，因此变化大的地方权重大，影响大
    # 我们希望神经网络关注的是整张图片变化，而不仅仅是重点关注一张图像中局部变化最大的部分
    # 因此除以方差之后，我们期望变化部分变得更加均匀，数据范围在-1～~+1或者-1.4到+1.4，不会出现一部分在10——20之间变，有的在0——255之间变

    return X


''' 2. 对训练数据集X_train中的每张图像做image_datagen处理（各种变换），然后显示变换后结果 '''
''' https://keras.io/zh/preprocessing/image/ '''
def show_samples_from_generator(image_datagen, X_train, y_train):
    # take a random image from the training set
    img_rgb = X_train[0]            # 从训练数据中找到第0张

    # plot the original image
    plt.figure(figsize=(1, 1))
    plt.imshow(img_rgb)
    plt.title('Example of RGB image (class = {})'.format(y_train[0]))
    plt.show()

    # plot some randomly augmented images
    rows, cols = 4, 10
    fig, ax_array = plt.subplots(rows, cols)
    for ax in ax_array.ravel():
        augmented_img, _ = image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1]).next()
        ax.imshow(np.uint8(np.squeeze(augmented_img)))
    '''numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能，区别：
        ○ ravel()：如果没有必要，不会产生源数据的副本;
        ○ flatten()：返回源数据的副本;
        ○ squeeze()：只能对维数为1的维度降维。 '''

    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.suptitle('Random examples of data augmentation (starting from the previous image)')     # 标题
    plt.show()

def show_image_generator_effect():
    X_train, y_train = load_traffic_sign_data('./traffic-signs-data/train.p')

    # Number of examples
    n_train = X_train.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # How many classes?
    n_classes = np.unique(y_train).shape[0]

    print("训练数据集的数据个数 = ", n_train)
    print("图像尺寸 = ", image_shape)
    print("类别数量 = ", n_classes)

    image_generatoe = get_image_generator()     # 只是定义了数据增强的手段、方式
    show_samples_from_generator(image_generatoe, X_train, y_train)




''' 3. 数据增强，从已有训练数据中产生更多的训练数据，对给定训练图像数据进行旋转 '''
def get_image_generator():
    # create the generator to perform online data augmentation
    image_datagen = ImageDataGenerator(rotation_range=15.,  # 旋转范围为15°，随机旋转
                                       zoom_range=0.2,      # 缩小0.2或者放大0.2
                                       width_shift_range=0.1,   # 图像在宽的方向上的挪动范围
                                       height_shift_range=0.1)  # 图像在高的方向上的挪动范围

    return image_datagen


''' 4. 获得训练模型 '''
def get_model(dropout_rate = 0.0):
    input_shape = (32, 32, 1)           # 训练数据集中每张图片是32×32，RGB通道变为YUV保留Y通道

    input = Input(shape=input_shape)
    cv2d_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(input)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_1)
    dropout_1 = Dropout(dropout_rate)(pool_1)
    flatten_1 = Flatten()(dropout_1)

    dense_1 = Dense(64, activation='relu')(flatten_1)   # 输出位64节点，激活函数为Relu
    output = Dense(43, activation='softmax')(dense_1)   # 输出为43节点，多元分类问题，激活函数用softmax
    model = Model(inputs=input, outputs=output)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                        # 使用交叉熵损失函数                                评价标准为准确率
    # summarize model
    model.summary()
    return model


''' 5. 模型训练 '''
def train(model, image_datagen, x_train, y_train, x_validation, y_validation):
    # checkpoint                          标签        验证数据        验证数据标签
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
                                        # 这里只存在验证数据集x_validation上准确率accuracy最大的模型

    callbacks_list = [checkpoint]
    image_datagen.fit(x_train)          # 这里要结合keras中ImageDataGenerator类来看是否需要加fit

    history = model.fit_generator(image_datagen.flow(x_train, y_train, batch_size=128),
                        steps_per_epoch=5000,
                        validation_data=(x_validation, y_validation),
                        epochs=8,
                        callbacks=callbacks_list,
                        verbose=1)
    '''
                        verbose：日志显示
                        verbose = 0
                        为不在标准输出流输出日志信息
                        verbose = 1
                        为输出进度条记录
                        verbose = 2
                        为每个epoch输出一行记录
                        注意： 默认为1
    '''

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])        # 第二个history是字典，包含acc准确率
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    with open('/trainHistoryDict.p', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    return history


''' 6. 模型评估 '''
def evaluate(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=1)
    accuracy = score[1]
    return accuracy


''' 7. 开始训练，主函数调用 '''
def train_model():
    X_train, y_train = load_traffic_sign_data('./traffic-signs-data/train.p')

    # Number of examples
    n_train = X_train.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # How many classes?
    n_classes = np.unique(y_train).shape[0]

    print("Number of training examples =", n_train)
    print("Image data shape  =", image_shape)
    print("Number of classes =", n_classes)

    X_train_norm = preprocess_features(X_train)             # 数据预处理  ''' 1. 图像预处理 '''
    y_train = utils.to_categorical(y_train, n_classes)      # 0-42，共43个整数来表示, 但训练的时候要用one-hot编码代替整数标签
    '''
    自然状态码为：000,001,010,011,100,101
    独热编码为：000001,000010,000100,001000,010000,100000
    '''

    # split into train and validation
    VAL_RATIO = 0.2
    X_train_norm, X_val_norm, y_train, y_val = train_test_split(X_train_norm, y_train,
                                                                test_size=VAL_RATIO,
                                                                random_state=0)

    model = get_model(0.0)                      # ''' 4. 获得训练模型 '''
    image_generator = get_image_generator()     # ''' 3. 数据增强，对给定训练图像数据进行旋转 '''
    train(model, image_generator, X_train_norm, y_train, X_val_norm, y_val)     # ''' 5. 模型训练 '''



if __name__ == "__main__":
    show_image_generator_effect()
    # train_model()



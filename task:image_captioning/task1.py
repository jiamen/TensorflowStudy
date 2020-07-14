
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPool2D


def generate_vgg16():
    '''
    搭建VGG16网络结构
    :return: VGG 16 网络结构
    '''
    input_shape = (224, 224, 3)
    model = Sequential([
        Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(1000, activation='softmax'),
    ])                          # VGG没有分叉, 所以是线性结构, Sequential是list结构

    return model







if __name__ == '__main__':
    model = generate_vgg16()
    model.summary()




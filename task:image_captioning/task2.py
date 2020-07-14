
from keras.models import model_from_json
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from pickle import dump
from os import listdir
import os
from keras.models import Model
import keras
from tqdm import tqdm


def load_img_as_np_array(path, target_size):
    '''
    从给定文件加载图像, 转换图像大小给定target_size, 返回Keras支持的浮点数numpy数组
    # Arguments:
        :param path: 图像文件路径
        :param target_size: 元组(图像高度, 图像宽度).
    # return:
        numpy 数组
    '''
    img = pil_image.open(path)                          # 打开文件
    img = img.resize(target_size, pil_image.NEAREST)    # NEAREST 是一种插值方法
    return np.asarray(img, dtype=K.floatx())            # 转化为向量


def preprocess_input(x):
    '''
    预处理图像用于网络输入, 将图像由RGB格式转换为BGR格式
    将图像的每一个图像通道减去均值
    均值BRG三个通道的均值分别为 103.939, 116.779, 123.68
    # Arguments:
        :param x: numpy 数组, 4维
    # return:
        Preprocessed Numpy array.
    '''
    # 'RGB' -> 'BGR', https://www.scivision.co/numpy-image-bgr-to-rgb/
    x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]

    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]

    return x


'''
总体步骤:
    提取图像的特征(利⽤VGG16的修改模型)
    初始化图像标题为”startseq”
    循环如下步骤:
        将图像标题转换为整数数组,每⼀个标题的单词对应于唯⼀⼀个整数
        将图像特征和当前的图像标题作为输⼊, 预测标题的下⼀个单词, 假设单词为word1
        将word1添加到当前标题的结尾
        如果word1的值为”endseq”, 或者当前标题的⻓度达到了标题最⼤⻓度, 退出循环
        此刻的图像标题就是预测的值
'''

''' 1. 从给定的VGG16⽹络结构⽂件和⽹络权值⽂件, 创建VGG16⽹络 '''
def load_vgg16_model():
    """从当前目录下面的 vgg16_exported.json 和 vgg16_exported.h5 两个文件中导入 VGG16 网络并返回创建的网络模型
    # Returns
        创建的网络模型 model
    """
    json_file = open("./Model/vgg16_exported.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()

    model=model_from_json(loaded_model_json)
    model.load_weights("./Model/vgg16_exported.h5")

    return model


def extract_features(directory):
    """提取给定文件夹中所有图像的特征, 将提取的特征保存在文件features.pkl中,
       提取的特征保存在一个dict中, key为文件名(不带.jpg后缀), value为特征值[np.array]
    Args:
        directory: 包含jpg文件的文件夹
    Returns:
        None
    """
    ''' 2. 修改⽹络结构(去除最后⼀层) '''
    model = load_vgg16_model()
    # pop the last layer
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print('Extracting...')

    ''' 3. 利⽤修改的⽹络结构,提取flicker8k数据集中所有图像的特征,使⽤字典存储, key为⽂件名, value为⼀个⽹络的输出。 '''
    features = dict()
    pbar = tqdm(total=len(listdir(directory)), desc="进度", ncols=100)
    for fn in listdir(directory):
        # print('\tRead File: ', fn)
        fn=directory+'/'+fn
        # 返回长、宽、通道的三维张量
        arr=load_img_as_np_array(fn, target_size=(224, 224))

        # 改变数组的形态，增加一个维度（批处理输入的维度）— — 4维
        arr=arr.reshape((1, arr.shape[0], arr.shape[1], arr.shape[2]))

        # 预处理图像作为VGG模型的输入
        arr = preprocess_input(arr)

        # 计算特征
        feature =model.predict(arr, verbose=0)

        # 分离文件名和路径以及分离文件名和后缀
        (filepath, tempfilename) = os.path.split(fn)
        (filename, extension) = os.path.splitext(tempfilename)
        id=tempfilename
        # print(id)
        features[id]=feature

    return features


''' 4. 将字典保存为features.pkl⽂件(使⽤pickle库) '''
if __name__ == '__main__':
    # 提取Flicker8k数据集中所有图像的特征，保存在一个文件中, 大约一小时的时间，最后的文件大小为127M
    # 下载zip文件，解压缩到当前目录的子文件夹Flicker8k_Dataset， 注意上传完成的作业时不要上传这个数据集文件
    directory = './DataSets/Flicker8k_Dataset'
    features = extract_features(directory)
    print('提取特征的文件个数：%d' % len(features))
    print(keras.backend.image_data_format())
    # 保存特征到文件
    dump(features, open('features.pkl', 'wb'))



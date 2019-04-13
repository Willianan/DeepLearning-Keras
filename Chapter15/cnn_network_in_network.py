# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/8 10:38
@Project:DeepLearning-Keras
@Filename:cnn_network_in_network.py
"""

"""
CNN -- Network In Network（池化层采用GlobalAveragePooling）
(01)、卷积层，具有192个特征图，感受野大小为3x3
(02)、卷积层，具有160个特征图，感受野大小为1x1
(03)、卷积层，具有96个特征图，感受野大小为1x1
(04)、采样因子（pool_size）为3x3，步长为2x2的池化层
(05)、Dropout层，Dropout概率为20%
(06)、卷积层，具有192个特征图，感受野大小为5x5
(07)、卷积层，具有192个特征图，感受野大小为1x1
(08)、卷积层，具有192个特征图，感受野大小为1x1
(09)、采样因子（pool_size）为3x3，步长为2x2的池化层
(10)、Dropout层，Dropout概率为50%
(11)、卷积层，具有192个特征图，感受野大小为5x5
(12)、卷积层，具有192个特征图，感受野大小为1x1
(13)、卷积层，具有10个特征图，感受野大小为1x1
(14)、使用GlobalAveragePooling作为最后一个池化层
(15)、激活层，使用激活函数softmax
"""



import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.initializers import RandomNormal
from keras import optimizers
from keras.callbacks import LearningRateScheduler, TensorBoard

batch_size = 128
epochs = 200
iterations = 391
num_classes = 10
dropout = 0.5
log_filepath = './nin'


def normalize_preprocessing(x_train, x_validation):
    x_train = x_train.astype('float32')
    x_validation = x_validation.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_validation[:, :, :, i] = (x_validation[:, :, :, i] - mean[i]) / std[i]

    return x_train, x_validation


def scheduler(epoch):
    if epoch <= 60:
        return 0.05
    if epoch <= 120:
        return 0.01
    if epoch <= 160:
        return 0.002
    return 0.0004


def build_model():
    model = Sequential()

    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001),
                     kernel_initializer=RandomNormal(stddev=0.01), input_shape=x_train.shape[1:],
                     activation='relu'))
    model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001),
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001),
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Dropout(dropout))

    model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001),
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001),
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001),
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Dropout(dropout))

    model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001),
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001),
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))
    model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001),
                     kernel_initializer=RandomNormal(stddev=0.05), activation='relu'))

    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))

    sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    np.random.seed(seed=7)
    # load data
    (x_train, y_train), (x_validation, y_validation) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_validation = keras.utils.to_categorical(y_validation, num_classes)

    x_train, x_validation = normalize_preprocessing(x_train, x_validation)

    # build network
    model = build_model()
    print(model.summary())

    # set callback
    tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]

    '''
    # set data augmentation
    print('Using real-time data augmentation.')
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=0.125, height_shift_range=0.125,
                                 fill_mode='constant', cval=0.)
    datagen.fit(x_train)

    # start training
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=iterations,
                        epochs=epochs, callbacks=cbks, validation_data=(x_validation, y_validation), verbose=2)
    '''
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=cbks,
              validation_data=(x_validation, y_validation), verbose=2)
    model.save('nin.h5')
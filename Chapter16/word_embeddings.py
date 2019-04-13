# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/8 10:51
@Project:DeepLearning-Keras
@Filename:word_embeddings.py
"""

# 词嵌入
# 潜入层需要指定词汇大小预期的最大数量，以及输出的每个词向量的维度。

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding

(x_train, y_train), (x_validation, y_validation) = imdb.load_data(num_words=5000)

x_train = sequence.pad_sequences(x_train, maxlen=500)
x_validation = sequence.pad_sequences(x_validation, maxlen=500)

Embedding(5000, 32, input_length=500)

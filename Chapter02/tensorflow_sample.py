# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/1 9:41
@Project:DeepLearning-Keras
@Filename:tensorflow_sample.py
"""

import tensorflow as tf

# 声明两个占位符
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# 定义表达式
add = tf.add(a, b)

# 执行运算
session = tf.Session()
binding = {a : 1.5, b : 2.5}
c = session.run(add, feed_dict=binding)
print(c)
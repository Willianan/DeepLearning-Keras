# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/1 9:43
@Project:DeepLearning-Keras
@Filename:theano_sample.py
"""

import theano
from theano import tensor

# 声明两个floating-point的占位符
a = tensor.dscalar()
b = tensor.dscalar()

# 创建一个表达式
c = a + b

# 将表达式编译到函数
f = theano.function([a, b], c)

# 执行这个函数
result = f(1.5, 2.5)
print(result)
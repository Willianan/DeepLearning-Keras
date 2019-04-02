# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/1 9:40
@Project:DeepLearning-Keras
@Filename:cntk_sample.py
"""

import cntk
a = [1, 2, 3]
b = [4, 5, 6]
c = cntk.minus(a, b).eval()
print(c)
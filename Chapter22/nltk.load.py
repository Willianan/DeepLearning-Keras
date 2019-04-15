# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/13 10:28
@Project:DeepLearning-Keras
@Filename:nltk.load.py
"""


import nltk
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# 下载nltk数据包
nltk.download()
# -*- coding:utf-8 -*-
"""
@Author：Charles Van
@E-mail:  williananjhon@hotmail.com
@Time：2019/4/9 9:19
@Project:DeepLearning-Keras
@Filename:load_ data.py
"""


from pandas import read_csv
from matplotlib import pyplot as plt

# 导入数据
fileName = "international-airline-passengers.csv"
data = read_csv(fileName,usecols=[1],engine='python',skipfooter=3)

# 图表表示
plt.plot(data)
plt.show()

# 查看最初的5条记录
print(data.head(5))
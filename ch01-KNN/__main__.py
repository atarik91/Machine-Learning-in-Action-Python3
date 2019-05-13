# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 21:18:57 2018

@author: Administrator
"""

# from numpy import *
# from Digit_recog import *
from Digit_recog import img2vec
from KNN_algr import *
from Parse_data import file_parse_matrix
from TestData import createDataSet, Test_accuray, Norm_feature
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 教材2.1节代码
    #group, labels = createDataSet()
    #output = classify_KNN([0, 0], group, labels, 3)
    #print("测试数据为:", [0, 0], "分类结果为：", output)

    # 教材2.2节代码
    DataMat, LabelMat = file_parse_matrix('datingTestSet2.txt')
    print(DataMat, shape(DataMat), LabelMat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(DataMat[:,1],DataMat[:,2])
    plt.show()

    dating_mat, label_mat = file_parse_matrix('datingTestSet2.txt')
    data_normed, ranges, minV = Norm_feature(dating_mat)
    Test_accuray(0.1, dating_mat, label_mat)


    # 教材2.3节代码
# testVec = img2vec('digits/testDigits/0_13.txt')
# print(testVec)

# HandWritingTest('digits/trainingDigits', 'digits/testDigits/')
# 这行代码耗时比较久，可以单独运行

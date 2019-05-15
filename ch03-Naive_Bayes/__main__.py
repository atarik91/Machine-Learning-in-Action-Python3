# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:10:01 2018

@author: Administrator
"""

from word_vector import *
from Train_bayes import *
from classifier_test import *
from Mail_filter import *
import feedparser
from RSS_filter import *

if __name__ == "__main__":

    wordlist, classlist = loadDataSet()
    
    myVoc = Create_wordVec(wordlist)
    print(myVoc)
    print(Words2Vec(myVoc, wordlist[0]))
    
    train_matrix = []
    for doc in wordlist:
        train_matrix.append(Words2Vec(myVoc, doc))
    
    print(train_matrix)
    p1,p2,p_pos = train_bayes(train_matrix, classlist)
    print(p1,p2,p_pos)
    
    Test_classify()

    #4.6--示例:使用朴素贝叶斯过滤垃圾邮件
    Spam_filter('email')

    '''4.7--基于概率论的分类方法:朴素贝叶斯示例'''
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    a, b, c = localWords(ny,sf)
    print(a,b,c)
    
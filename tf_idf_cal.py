# encoding: utf-8
import sys
import jieba.posseg as pseg
import jieba
import jieba.analyse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
def features():
    # load comments
    df = pd.read_csv('data/comments.csv')

    #complete the missing datas with NULL
    df.comments = df.comments.fillna('')

    #select the comments having '相信'
    res = [i for i in df.comments if '相信' in i ]

    #delete '回复' from all selected comments
    res = [i.lstrip('回复') for i in res ]
    #print len(res)

    # words cut with jieba
    result = []
    for i in res:
        result.append(''.join(jieba.cut(i)))
    print('$$$$$$$$')
    print(result)
    print('$$$$$$$$')

    #compute the values of word with tf-idf
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf = vectorizer.fit_transform(result)

    #print sorted valued words
    words = vectorizer.get_feature_names()

    return tfidf
    #print len(words)

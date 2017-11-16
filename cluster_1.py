#! python3
# -*- coding: utf-8 -*-
import json
import jieba
from random import shuffle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
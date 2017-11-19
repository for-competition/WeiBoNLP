# WeiBoNLP
Sentiment Analysis: 微博语义情感分析
# Cluster
https://weibo.com/5861369000/FlCRqC4wl?filter=hot&root_comment_id=0&type=comment#_rnd1505480063572
1.提取上面链接的评论
2.预处理：去除用户名，回复之类的噪音文本
3.先jieba分词，再去除停用词
4.利用sklearn框架进行聚类，主要通过CountVectorizer()以及TfidfTransformer()，CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵。
  矩阵元素weight[i][j] 表示j词在第i个文本下的词频，即各个词语出现的次数。TfidfTransformer也有个fit_transform函数，它的作用是计算tf-idf值。
5.利用Kfold进行K折交叉验证，K折交叉验证会自动划分训练集和验证集，此举是为了在数据集比较小的情况下，让聚类效果趋于稳定
6.求Kfold的平均距离，确定平均参数
7.利用matplotlib画error图，确定最好的聚类中心簇个数
8.确定中心簇的个数为5，得到不同簇的词语
# Todo
1.由于聚类效果不好，并不是二元结果，即根据评论判断用户对婚姻持肯定还是否定态度，改变策略
2.人工对评论打正向负向标签
3.svm进行二元分类学习
4.区分语境，例如：相信不与不相信，虽然tf-idf特征一样，但并不是相同的含义

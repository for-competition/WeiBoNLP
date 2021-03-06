# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#! python3
# -*- coding: utf-8 -*-
import os
import json
import jieba
import logging
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from scipy.sparse.csr import csr_matrix
from sklearn.model_selection import KFold


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger()


class Prepare(object):
    def __init__(self, filename):
        self.filename = filename

    def extract(self):
        """
        提取json数据中的原始文本
        :return:
        """
        with open(self.filename, 'rb') as f:
            data = json.loads(f.read())
        record = data['RECORDS']
        comment = []
        for item in record:
            logging.info(item['id'])
            single_comment = item['comment_content']
            comment.append(single_comment)
            # break
        return comment

    def remove_noise_text(self, raw_comment):
        """
        去除用户名，回复噪音文本
        :return:
        """
        comment = []
        for item in raw_comment:
            ret = item.split(u'：')[-1]
            if ':' in ret:
                real_comment = ret.split(':')[-1]
                logger.info(real_comment)
                comment.append(real_comment)
            else:
                comment.append(ret)
        comment = [i for i in comment if i != ""]
        return comment

    def cut_word(self, comment):
        """
        顺序：先分词，去除停用词
        去除停用词的方法参考：https://www.zhihu.com/question/41199317
        比较好的方法是用extract_tags函数，这个函数会根据TF-IDF算法将特征词提取出来，
        在提取之前会去掉停用词，可以人工指定停用词字典
        jieba.analyse.set_stop_words('D:\\Python27\\stopword.txt')
        tags = jieba.analyse.extract_tags(text,20)
        :return:
        """
        all_word = []
        for item in comment:
            ret = list(jieba.cut(item, cut_all=False))
            all_word.extend(ret)
        corpus = [i for i in all_word if i != " " and i != "龙卷风" and i != "嘴里塞"]   # 将龙卷风，嘴里塞加入停用词表并没用
        # analyse.set_stop_words('D:\\Python36 Project\\WuHanNLP_Dev\\stop_word\\所有停用词.txt')
        # result = analyse.extract_tags(all_word, topK=1000, withWeight=True, allowPOS=())
        # no_stop_word = []
        # for item in result:
        #     logger.info(item[0])
        #     no_stop_word.append(item[0])
        # return no_stop_word
        return corpus

    def cluster(self, corpus):
        """
        开始聚类
        sklearn里面的TF-IDF主要用到了两个函数：CountVectorizer()和TfidfTransformer()。
        CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵。
        矩阵元素weight[i][j] 表示j词在第i个文本下的词频，即各个词语出现的次数。
        通过get_feature_names()可看到所有文本的关键字，通过toarray()可看到词频矩阵的结果。
        TfidfTransformer也有个fit_transform函数，它的作用是计算tf-idf值。
        :param corpus:
        :return:
        """
        base_path = os.path.abspath(os.path.dirname(__file__))
        stop_word_path = os.path.join(base_path, 'stop_word')
        stop_word_list = []
        with open('%s/%s.txt' % (stop_word_path, '所有停用词'), 'r', encoding='utf8') as f:
            for line in f.readlines():
                stop_word_list.append(line.replace('\n', ''))
        print(stop_word_list)
        stop_word_set = frozenset(stop_word_list)
        """
        指定停用词
        """
        vectorizer = CountVectorizer(stop_words=stop_word_set)
        transformer = TfidfTransformer()
        tf_idf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        kf = KFold(n_splits=10)
        result = []
        for i in range(2, 11):
            logger.info(i)
            km = KMeans(n_clusters=i)
            cluster_result = []
            for train_index, test_index in kf.split(tf_idf):
                X_train = tf_idf[train_index]
                X_test = tf_idf[test_index]
                km.fit(X_train)
                logger.info(km.inertia_)
                cluster_result.append({'score': -km.score(X_test), 'cluster_num': i})
            result.append(cluster_result)
        return result
    
    def average(self, result):
        """
        对k_fold的距离求平均距离
        """
        length = len(result)
        average_cluster_result = []
        for index in range(length):
            matching_cluster_data = result[index]
            init = 0
            for item in matching_cluster_data:
                init += item['score']
            average_distance = init / 10
            logger.info(average_distance)
            cluster_num = index + 2
            average_cluster_result.append({'score': average_distance, 'cluster_num': cluster_num})
        return average_cluster_result
    
    def draw(self, result):
        """
        画error图
        :param result:
        :return:
        """
        score = []
        true_ks = []
        for item in result:
            score.append(item['score'])
            true_ks.append(item['cluster_num'])
        plt.figure(figsize=(8, 4))
        plt.plot(true_ks, score, label='error', color='red', linewidth=1)
        plt.xlabel("n_features")
        plt.ylabel("error")
        plt.legend()
        plt.show()

    def get_label(self, corpus, n_cluster=5):
        """
        经过matplotlib作图可知最好的簇的个数为5
        :param corpus
        :param n_cluster:
        :return:
        """
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tf_idf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        train_x, test_x = train_test_split(tf_idf, test_size=0.2)
        km = KMeans(n_clusters=n_cluster)
        km.fit(train_x)
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print(vectorizer.get_stop_words())
        for i in range(n_cluster):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print('\n')


def main():
    prepare = Prepare('comment.json')
    raw_comment = prepare.extract()
    comment = prepare.remove_noise_text(raw_comment)
    no_stop_word = prepare.cut_word(comment)
    all_cluster_result = prepare.cluster(no_stop_word)
    average_cluster_result = prepare.average(all_cluster_result)
    print(average_cluster_result)
    prepare.draw(average_cluster_result)
    prepare.get_label(no_stop_word)
    # with open('no_stop_word.json', 'w', encoding='utf8') as f:
    #     f.write(json.dumps(no_stop_word, sort_keys=True, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()


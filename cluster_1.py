#! python3
# -*- coding: utf-8 -*-
import json
import jieba
import logging
from jieba import analyse
from random import shuffle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


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
        all_word = ' '.join([i for i in all_word if i != " "])
        analyse.set_stop_words('D:\\Python36 Project\\WuHanNLP_Dev\\stop_word\\所有停用词.txt')
        result = analyse.extract_tags(all_word, topK=1000, withWeight=True, allowPOS=())
        no_stop_word = []
        for item in result:
            logger.info(item[0])
            no_stop_word.append(item[0])
        return no_stop_word

    def cluster(self, corpus):
        """
        开始聚类
        :param corpus:
        :return:
        """
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tf_idf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        train_x, test_x = train_test_split(tf_idf, test_size=0.2)
        cluster_result = []
        for i in range(2, 11):
            km = KMeans(n_clusters=i)
            km.fit(train_x)
            print(km.inertia_)
            cluster_result.append({'score': -km.score(test_x), 'cluster_num': i})
        return cluster_result

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


def main():
    prepare = Prepare('comment.json')
    raw_comment = prepare.extract()
    comment = prepare.remove_noise_text(raw_comment)
    no_stop_word = prepare.cut_word(comment)
    cluster_result = prepare.cluster(no_stop_word)
    prepare.draw(cluster_result)
    # with open('no_stop_word.json', 'w', encoding='utf8') as f:
    #     f.write(json.dumps(no_stop_word, sort_keys=True, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()


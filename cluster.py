#! python3
# -*- coding: utf-8 -*-
"""
目标：实现小规模文本的聚类
1.去除无用词
2.分词
3.tf-idf
4.聚类
"""
import json
import jieba
from random import shuffle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
"""
sklearn里面的TF-IDF主要用到了两个函数：CountVectorizer()和TfidfTransformer()。 
CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵。 
矩阵元素weight[i][j] 表示j词在第i个文本下的词频，即各个词语出现的次数。 
通过get_feature_names()可看到所有文本的关键字，通过toarray()可看到词频矩阵的结果。 
TfidfTransformer也有个fit_transform函数，它的作用是计算tf-idf值。
"""
def extract_native_comment(filename):
    """
    从数据库中提取原始文本
    :param filename:
    :return:
    """
    with open(filename, 'rb') as f:
        data = json.loads(f.read())
        record = data['RECORDS']
        comment = []
        for item in record:
            # logger.info(item['id'])
            single_comment = item['comment_content']
            # print(single_comment)
            comment.append(single_comment)
            # break
    # print(comment)
    return comment


def shuffle_comment(comment):
    """
    1.先打乱顺序
    2.将短文本进行中文冒号分割，提取真正的评论
    :param comment:
    :return:
    """
    shuffle(comment)
    real_comment = []
    for item in comment:
        print(item)
        split_list = item.split(u'：')
        print(split_list)
        length = len(split_list)
        if length == 1:
            real_comment.append(item)
        elif length > 1:
            flag = False
            for idx in range(1, length):
                if '回复' in split_list[idx]:
                    inner_split_list = split_list[idx].split(':')
                    print(inner_split_list[-1])
                    print('------------------')
                    real_comment.append(inner_split_list[-1])
                    flag = True
            if not flag:
                """
                如果遍历完成后，都没有回复的词语
                """
                real_comment.append(split_list[-1])
    print(real_comment)
    return real_comment


def cut_word(real_comment):
    after_cut_word = []
    for item in real_comment:
        print('----------------->')
        print(item)
        seg_list = list(jieba.cut(item, cut_all=False))  # 精确模式，适合文本分析
        print(seg_list)
        print('----------------->')
        after_cut_word.extend(seg_list)
        # break
    # print(after_cut_word)
    return after_cut_word


def remove_stop_word(after_cut_word):
    """
    去除停用词，暂时还是用的网上通用的停用词(如哈工大)，后面看效果可能会构建微博独有的停用词表
    :param after_cut_word:
    :return:
    """
    non_empty = [i for i in after_cut_word if i != ' ']  # 去除空白符
    stop_word_collection = []
    with open('stop_word/所有停用词.txt', 'r', encoding='utf8') as f:
        for line in f.readlines():
            if line != '\n':
                stop_word_collection.append(line)
                # print(line)
    after_remove = []
    for word in non_empty:
        if word not in stop_word_collection:
            after_remove.append(word)
    return after_remove


def continue_remove_useless_word(after_remove):
    """
    肉眼发现还是有一些无用词语，要进行是否是汉字的判断
    :param after_remove:
    :return:
    """
    final_result = [char for char in after_remove if '\u4e00' <= char <= '\u9fff']
    return final_result


def feature(corpus):
    """
    sklearn里面的TF-IDF主要用到了两个函数：CountVectorizer()和TfidfTransformer()。
    CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵。
    矩阵元素weight[i][j] 表示j词在第i个文本下的词频，即各个词语出现的次数。
    通过get_feature_names()可看到所有文本的关键字，通过toarray()可看到词频矩阵的结果。
    TfidfTransformer也有个fit_transform函数，它的作用是计算tf-idf值。
    """
    vectorizer = CountVectorizer()  # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tf_idf = transformer.fit_transform(vectorizer.fit_transform(corpus))
    print('%%%%%%%%%')
    print(tf_idf)
    print('%%%%%%%%%')
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    print('&&&&&&&&&')
    print(word)
    print('&&&&&&&&&')
    print('$$$$$$$$$')
    weight = tf_idf.toarray()  # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    print(weight)
    print('$$$$$$$$$')

    train_x, test_x = train_test_split(tf_idf, test_size=0.2)
    # scores = []
    # for i in range(2, 21):
    #     km = KMeans(n_clusters=i)
    #     km.fit(train_x)
    #     label = km.labels_
    #     print(label)
    #     print(km.inertia_)  # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选择临界点的簇的个数
    #     scores.append({-km.score(test_x): i})
    # 确定簇的个数
    # return 19

    km = KMeans(n_clusters=19)
    km.fit(train_x)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    print(vectorizer.get_stop_words())
    for i in range(19):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()

    # sort_score = sorted(scores, key=lambda k: k[0], reverse=True)
    # print(sort_score)


def main():
    comment = extract_native_comment('comment.json')
    real_comment = shuffle_comment(comment)
    after_cut_word = cut_word(real_comment)
    after_remove = remove_stop_word(after_cut_word)
    final_result = continue_remove_useless_word(after_remove)
    feature(final_result)
    # print(final_result)
    # with open('final_result.txt', 'w') as f:
    #     f.write(json.dumps(final_result, sort_keys=True, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()

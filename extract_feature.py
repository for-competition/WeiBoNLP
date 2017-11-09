#! python3
# -*- coding: utf-8 -*-
"""
特征工程
找到微博独有的停用词，而不是用网上的通用词表
1.先进行内容的关键词提取，使用jieba分词+tf-idf.也可以使用textrank来提取内容关键词
短文本可能采用tf-idf更好
"""
import re
import json
import jieba
import numpy as np
from random import shuffle
from logger.log import logger


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


def main():
    comment = extract_native_comment('comment.json')
    real_comment = shuffle_comment(comment)
    after_cut_word = cut_word(real_comment)
    after_remove = remove_stop_word(after_cut_word)
    final_result = continue_remove_useless_word(after_remove)
    with open('final_result.txt', 'w') as f:
        f.write(json.dumps(final_result, sort_keys=True, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()
    # url = 'http://finance.sina.com.cn/china/gncj/2017-03-20/doc-ifycnpit2359846.html'
    # domain_dir_name = re.findall(r'http://.+cn/', url)[0].replace('http://', '').replace('/', '')
    # print(domain_dir_name)

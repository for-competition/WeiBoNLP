#! python3
# -*- coding: utf-8 -*-
import json
import jieba
from jieba import analyse


def tf_idf(filename):
    analyse.set_stop_words('D:\\Python36 Project\\WuHanNLP_Dev\\stop_word\\1915stopwords.txt')
    with open(filename, 'rb') as f:
        data = json.loads(f.read())
        record = data['RECORDS']
        comment = []
        for item in record:
            print(item['id'])
            single_comment = item['comment_content']
            comment.append(single_comment)
    complete_text = ','.join(comment)
    seg = analyse.extract_tags(complete_text, topK=20, withWeight=True, allowPOS=())
    for tag, weight in seg:
        print("%s %s" % (tag, weight))


def main():
    tf_idf('comment.json')


if __name__ == '__main__':
    main()

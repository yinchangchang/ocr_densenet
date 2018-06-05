# coding=utf8
#########################################################################
# File Name: map_word_to_index.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: Fri 18 May 2018 03:30:26 PM CST
#########################################################################
'''
此代码用于将所有文字映射到index上，有两种方式
    1. 映射每一个英文单词为一个index
    2. 映射每一个英文字母为一个index
'''

import os
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import json
from collections import OrderedDict

def map_word_to_index(train_word_file, word_index_json, word_count_json, index_label_json, alphabet_to_index=True):
    with open(train_word_file, 'r') as f:
        labels = f.read().strip().decode('utf8')
    word_count_dict = { }
    for line in labels.split('\n')[1:]:
        line = line.strip()
        image, sentence = line.strip().split('.png,')
        sentence = sentence.strip('"')
        for w in sentence:
            word_count_dict[w] = word_count_dict.get(w,0) + 1
    print '一共有{:d}种字符，共{:d}个'.format(len(word_count_dict), sum(word_count_dict.values()))
    word_sorted = sorted(word_count_dict.keys(), key=lambda k:word_count_dict[k], reverse=True)
    # word_index_dict = { w:i for i,w in enumerate(word_sorted) }
    word_index_dict = json.load(open(word_index_json))

    with open(word_count_json, 'w') as f:
        f.write(json.dumps(word_count_dict, indent=4, ensure_ascii=False))
    # with open(word_index_json, 'w') as f:
    #     f.write(json.dumps(word_index_dict, indent=4, ensure_ascii=False))
        
    image_label_dict = OrderedDict()
    for line in labels.split('\n')[1:]:
        line = line.strip()
        image, sentence = line.strip().split('.png,')
        sentence = sentence.strip('"')

        # 换掉部分相似符号
        for c in u"　 ":
            sentence = sentence.replace(c, '')
        replace_words = [
                u'(（',
                u')）',
                u',，',
                u"´'′", 
                u"″＂“",
                u"．.",
                u"—-"
                ]
        for words in replace_words:
            for w in words[:-1]:
                sentence = sentence.replace(w, words[-1])

        index_list = []
        for w in sentence:
            index_list.append(str(word_index_dict[w]))
        image_label_dict[image + '.png'] = ' '.join(index_list)
    with open(index_label_json, 'w') as f:
        f.write(json.dumps(image_label_dict, indent=4))


def main():

    # 映射字母为index
    train_word_file = '../../files/train.csv'
    word_index_json = '../../files/alphabet_index_dict.json'
    word_count_json = '../../files/alphabet_count_dict.json'
    index_label_json = '../../files/train_alphabet.json'
    map_word_to_index(train_word_file, word_index_json, word_count_json, index_label_json, True)

if __name__ == '__main__':
    main()

# coding=utf8
#########################################################################
# File Name: analysis_dataset.py
# Author: ccyin
# mail: ccyin04@gmail.com
# Created Time: Fri 18 May 2018 04:19:58 PM CST
#########################################################################
'''
此文件用于分析原有数据集信息
    stati_image_size: 统计图片大小信息
    stati_label_length: 统计文字长度信息
'''

import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../ocr')
from tools import plot

def stati_image_size(image_dir, save_dir, big_w_dir):
    if not os.path.exists(big_w_dir):
        os.mkdir(big_w_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    h_count_dict, w_count_dict, r_count_dict = { }, { }, { }
    image_hw_ratio_dict = { }
    for image in os.listdir(image_dir):
        h,w = Image.open(os.path.join(image_dir, image)).size
        if w > 80:
            cmd = 'cp ../../data/dataset/train/{:s} {:s}'.format(image, big_w_dir)
            # os.system(cmd)

        r = int(h / 8. / w)
        h = h / 10
        w = w / 10
        r_count_dict[r] = r_count_dict.get(r, 0) + 1
        h_count_dict[h] = h_count_dict.get(h, 0) + 1
        w_count_dict[w] = w_count_dict.get(w, 0) + 1
        image_hw_ratio_dict[image] = r

    with open(os.path.join(save_dir, 'image_hw_ratio_dict.json'), 'w') as f:
        f.write(json.dumps(image_hw_ratio_dict, indent=4))

    x = range(max(h_count_dict.keys())+1)
    y = [0 for _ in x]
    for h in sorted(h_count_dict.keys()):
        print '图片长度:{:d}~{:d}，有{:d}张图'.format(10*h, 10*h+10, h_count_dict[h])
        y[h] = h_count_dict[h]
    plot.plot_multi_line([x], [y], ['Length'], save_path='../../data/length.png', show=True)

    x = range(max(w_count_dict.keys())+1)
    y = [0 for _ in x]
    for w in sorted(w_count_dict.keys()):
        print '图片宽度:{:d}~{:d}，有{:d}张图'.format(10*w, 10*w+10, w_count_dict[w])
        y[w] = w_count_dict[w]
    plot.plot_multi_line([x], [y], ['Width'], save_path='../../data/width.png', show=True)

    x = range(max(r_count_dict.keys())+1)
    y = [0 for _ in x]
    for r in sorted(r_count_dict.keys()):
        print '图片比例:{:d}~{:d}，有{:d}张图'.format(8*r, 8*r+8, r_count_dict[r])
        y[r] = r_count_dict[r]
    x = [8*(_+1) for _ in x]
    plot.plot_multi_line([x], [y], ['L/W'], save_path='../../data/ratio.png', show=True)

    print '\n最多的长\n', sorted(h_count_dict.keys(), key=lambda h:h_count_dict[h])[-1] * 10
    print '\n最多的宽\n', sorted(w_count_dict.keys(), key=lambda w:w_count_dict[w])[-1] * 10

    print '建议使用 64 * 512 的输入'
    print '    部分使用 64 * 1024 的输入'
    print '    剩下的忽略'
    print '建议使用FCN来做，全局取最大值得到最终结果'

def stati_label_length(label_json, long_text_dir):
    if not os.path.exists(long_text_dir):
        os.mkdir(long_text_dir)
    image_label_json = json.load(open(label_json))
    l_count_dict = { }
    for image, label in image_label_json.items():
        l = len(label.split())
        l_count_dict[l] = l_count_dict.get(l, 0) + 1
        if l > 25:
            cmd = 'cp ../../data/dataset/train/{:s} {:s}'.format(image, long_text_dir)
            # os.system(cmd)

    word_num = 0.
    x = range(max(l_count_dict.keys())+1)
    y = [0 for _ in x]
    for l in sorted(l_count_dict.keys()):
        word_num += l * l_count_dict[l]
        print '文字长度:{:d}，有{:d}张图'.format(l, l_count_dict[l])
        y[l] = l_count_dict[l]
    plot.plot_multi_line([x], [y], ['Word Number'], save_path='../../data/word_num.png', show=True)
    print '平均每张图片{:3.4f}个字'.format(word_num / sum(l_count_dict.values()))

def stati_image_gray(image_dir):
    print 'eval train image gray'
    for image in tqdm(os.listdir(image_dir)):
        image = Image.open(os.path.join(image_dir, image)).convert('RGB')
        image = np.array(image)
        mi,ma = image.min(), image.max()
        assert mi >= 0
        assert ma < 256

    print 'eval test image gray'
    image_dir = image_dir.replace('train', 'test')
    for image in tqdm(os.listdir(image_dir)):
        image = Image.open(os.path.join(image_dir, image)).convert('RGB')
        image = np.array(image)
        mi,ma = image.min(), image.max()
        assert mi >= 0
        assert ma < 256



def main():
    image_dir = '../../data/dataset/train'
    save_dir = '../../files/'
    big_w_dir = '../../data/big_w_dir'
    stati_image_size(image_dir, save_dir, big_w_dir)

    train_label_json = '../../files/train_alphabet.json'
    long_text_dir = '../../data/long_text_dir'
    stati_label_length(train_label_json, long_text_dir)
    # stati_image_gray(image_dir)

if __name__ == '__main__':
    main()

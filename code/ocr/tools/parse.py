# coding=utf8

import argparse

parser = argparse.ArgumentParser(description='medical caption GAN')

parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='densenet',
        help='model'
        )
parser.add_argument(
        '--data-dir',
        '-d',
        type=str,
        default='../../data/dataset/',
        help='data directory'
        )
parser.add_argument(
        '--bg-dir',
        type=str,
        default='../../data/images',
        help='back groud images directory'
        )
parser.add_argument(
        '--hard-mining',
        type=int,
        default=0,
        help='use hard mining'
        )
parser.add_argument('--phase',
        default='train',
        type=str,
        metavar='S',
        help='pretrain/train/test phase')
parser.add_argument(
        '--batch-size',
        '-b',
        metavar='BATCH SIZE',
        type=int,
        default=16,
        help='batch size'
        )
parser.add_argument('--save-dir',
        default='../../data',
        type=str,
        metavar='S',
        help='save dir')
parser.add_argument('--word-index-json',
        default='../../files/alphabet_index_dict.json',
        type=str,
        metavar='S',
        help='save dir')
parser.add_argument('--black-json',
        default='../../files/black.json',
        type=str,
        metavar='S',
        help='black_list json')
parser.add_argument('--image-hw-ratio-json',
        default='../../files/image_hw_ratio_dict.json',
        type=str,
        metavar='S',
        help='image h:w ratio dict')
parser.add_argument('--word-count-json',
        default='../../files/alphabet_count_dict.json',
        type=str,
        metavar='S',
        help='word count file')
parser.add_argument('--image-label-json',
        default='../../files/train_alphabet.json',
        type=str,
        metavar='S',
        help='image label json')
parser.add_argument('--resume',
        default='',
        type=str,
        metavar='S',
        help='start from checkpoints')
parser.add_argument('--no-aug',
        default=0,
        type=int,
        metavar='S',
        help='no augmentation')
parser.add_argument('--small',
        default=1,
        type=int,
        metavar='S',
        help='small fonts')
parser.add_argument('--difficult',
        default=0,
        type=int,
        metavar='S',
        help='只计算比较难的图片')
parser.add_argument('--hist',
        default=0,
        type=int,
        metavar='S',
        help='采用直方图均衡化')
parser.add_argument('--feat',
        default=0,
        type=int,
        metavar='S',
        help='生成LSTM的feature')

#####
parser.add_argument('-j',
        '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 32)')
parser.add_argument('--lr',
        '--learning-rate',
        default=0.001,
        type=float,
        metavar='LR',
        help='initial learning rate')
parser.add_argument('--epochs',
        default=10000,
        type=int,
        metavar='N',
        help='number of total epochs to run')
parser.add_argument('--save-freq',
        default='5',
        type=int,
        metavar='S',
        help='save frequency')
parser.add_argument('--save-pred-freq',
        default='10',
        type=int,
        metavar='S',
        help='save pred clean frequency')
parser.add_argument('--val-freq',
        default='5',
        type=int,
        metavar='S',
        help='val frequency')
parser.add_argument('--debug',
        default=0,
        type=int,
        metavar='S',
        help='debug')
parser.add_argument('--input-filter',
        default=7,
        type=int,
        metavar='S',
        help='val frequency')
parser.add_argument('--use-gan',
        default=0,
        type=int,
        metavar='S',
        help='use GAN')
parser.add_argument('--write-pred',
        default=0,
        type=int,
        metavar='S',
        help='writ predictions')
parser.add_argument(
        '--result-file',
        '-r',
        type=str,
        default='../../data/result/test_result.csv',
        help='result file'
        )
parser.add_argument(
        '--output-file',
        '-o',
        type=str,
        default='../../data/result/test.csv',
        help='output file'
        )
args = parser.parse_args()

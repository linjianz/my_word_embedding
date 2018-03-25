#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-16 22:17:47
Program: 
Description: 
"""
import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from preprocess.word_process import PreProcess
from preprocess.huffman_tree import HuffmanTree
from utils.misc import normalize, sigmoid


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vec_len',         default=300, type=int, help='vector length')
    parser.add_argument('--min_freq',        default=60, type=int, help='min frequency')

    return parser.parse_args()


class Word2Vec(object):
    def __init__(self, text, word_dict, vec_len=150, lr=0.01, win_len=5, model='cbow'):
        self.text_list = text
        self.word_dict = word_dict
        self.vec_len = vec_len
        self.lr = lr
        self.win_len = win_len
        self.model = model
        self.huffman = HuffmanTree(self.word_dict, vec_len=self.vec_len)

    def train(self):
        print('Start Training'.center(70, '='))
        before = (self.win_len-1) >> 1  # 2
        after = self.win_len-1-before   # 2

        if self.model == 'cbow':
            method = self.update_use_cbow
        else:
            method = self.update_use_skip_gram

        for l in tqdm(range(len(self.text_list))):
            line = self.text_list[l]
            line_len = len(line)
            for i in range(line_len):
                method(line[i], line[max(0, i - before):i] + line[i + 1:min(line_len, i + after + 1)])

    def update_use_cbow(self, word, word_list):
        """
        word: 当前单词
        word_list: 周围的单词
        """
        if not self.word_dict.__contains__(word):
            return

        huffman_code = self.word_dict[word]['Huffman']  # eg: 001
        vector_sum = np.zeros([1, self.vec_len])
        for i in np.arange(len(word_list))[::-1]:  # i: 3,2,1,0
            item = word_list[i]
            if self.word_dict.__contains__(item):  # 判断有没有这个词，有就加上vector，没有就pop
                vector_sum += self.word_dict[item]['vector']
            else:
                word_list.pop(i)

        if len(word_list) == 0:  # 如果周围的词都没有，就直接返回
            return

        e = self.go_along_huffman(huffman_code, vector_sum, self.huffman.root)

        for item in word_list:
            self.word_dict[item]['vector'] += e
            self.word_dict[item]['vector'] = normalize(self.word_dict[item]['vector'])

    def update_use_skip_gram(self, word, word_list):

        if not self.word_dict.__contains__(word):
            return

        word_vector = self.word_dict[word]['vector']
        for i in np.arange(len(word_list))[::-1]:
            if not self.word_dict.__contains__(word_list[i]):
                word_list.pop(i)

        if len(word_list) == 0:
            return

        for u in word_list:
            u_huffman = self.word_dict[u]['Huffman']
            e = self.go_along_huffman(u_huffman, word_vector, self.huffman.root)
            self.word_dict[word]['vector'] += e
            self.word_dict[word]['vector'] = normalize(self.word_dict[word]['vector'])

    def go_along_huffman(self, word_huffman, input_vector, root):
        """
        :param word_huffman:    当前词的huffman编码，例如：001
        :param input_vector:    周围单词的词向量的和
        :param root:            根节点
        :return:                当前单词的梯度变化总和
        """
        node = root
        e = np.zeros([1, self.vec_len])
        for level in range(word_huffman.__len__()):             # 从huffman tree上往下搜索
            huffman_char = word_huffman[level]                  # 0 或者 1
            q = sigmoid(input_vector.dot(node.value.T))         # 为什么不是average后再点乘？？？
            grad = self.lr * (1-int(huffman_char)-q)            # 计算梯度
            e += grad * node.value                              # 累加当前词的梯度变化
            node.value += grad * input_vector                   # 跟新结点的向量
            node.value = normalize(node.value)
            if huffman_char == '0':
                node = node.right
            else:
                node = node.left
        return e

    def save_word_vector(self):
        dir_save = './data'
        if not os.path.exists(dir_save):
            os.mkdir(dir_save)
        with open(dir_save+'/word_vector.pkl', 'wb') as f:
            pickle.dump(self.word_dict, f)
        word_vector_norm = {w: normalize(self.word_dict[w]['vector']) for w in self.word_dict.keys()}
        with open(dir_save+'/word_vector_norm.pkl', 'wb') as f:
            pickle.dump(word_vector_norm, f)


if __name__ == '__main__':
    parser_args = get_parser()
    dir_data = '/media/csc105/Data/dataset-jiange/nlp'
    pp = PreProcess(dir0=dir_data, min_freq=parser_args.min_freq, vec_len=parser_args.vec_len, sen_len=50)
    wv = Word2Vec(text=pp.text,
                  word_dict=pp.word_dict,
                  vec_len=parser_args.vec_len,
                  lr=0.01,
                  win_len=5,
                  model='cbow')
    wv.train()
    wv.save_word_vector()
    print('\n')
    print('END'.center(70, '='))

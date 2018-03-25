#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-19 22:42:55
Program: 
Description: 
"""
import numpy as np
from tqdm import tqdm


class PreProcess(object):
    def __init__(self, dir0, min_freq, vec_len, sen_len=50):
        self.dir_data = dir0
        self.min_freq = min_freq
        self.vec_len = vec_len
        self.sen_len = sen_len
        print('\n')
        print('Generate Word Dictionary'.center(70, '='))
        self.text, self.word_freq = self.word_count()
        self.word_dict = self.get_word_dict()

    def word_count(self):
        """
        根据文本统计词频、剔除低频词，返回word_freq {word: freq}
        """
        with open(self.dir_data+'/text8') as f:
            text = f.read()
        words = text.split()  # list

        # 分句
        sen_count = len(words)//self.sen_len
        text_list = [words[i*self.sen_len: (i+1)*self.sen_len] for i in range(sen_count)]
        if sen_count * self.sen_len < len(words):
            text_list.append(words[sen_count*self.sen_len:])

        # 统计词频
        dict_freq = dict()
        for word in tqdm(words):
            if word not in dict_freq:
                dict_freq[word] = 1
            else:
                dict_freq[word] += 1
        print('Sentences:\t\t{}'.format(sen_count))
        print('Total words:\t{}\nVocabulary:\t\t{}'.format(len(words), len(dict_freq)))     # 170w, 25w

        # 剔除低频词
        dict_freq_filtered = {word: freq for word, freq in dict_freq.items() if freq >= self.min_freq}
        print('After filter:\t{}'.format(len(dict_freq_filtered)))                          # 7w

        # 剔除停用词
        # with open(self.dir_data+'/stop_words.pkl', 'rb') as f:
        #     stop_words = pickle.load(f)
        #
        # for word in stop_words:
        #     try:
        #         dict_freq_filtered.pop(word)
        #     except:
        #         pass

        return text_list, dict_freq_filtered

    def get_word_dict(self):
        """
        生成词典 word_dict {word: {word, freq, possibility, vector, huffman}}
        """
        word_dict = dict()
        sum_count = sum(self.word_freq.values())
        for word, freq in self.word_freq.items():
            temp_dict = dict(
                word=word,
                freq=freq,
                possibility=freq/sum_count,
                vector=np.random.random([1, self.vec_len]),
                Huffman=None
            )
            word_dict[word] = temp_dict
        return word_dict


if __name__ == '__main__':
    dir_data = '/media/csc105/Data/dataset-jiange/nlp'
    pp = PreProcess(dir0=dir_data, min_freq=10, vec_len=150, sen_len=50)
    word_dict = pp.word_dict
    print(len(word_dict))
    count = 0
    for value in word_dict.values():
        print(value['word'], value['possibility'])
        count += 1
        if count == 10:
            break

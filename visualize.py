#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-22 21:16:56
Program: 
Description: 
"""
import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_with_labels(low_dim_embs, labels, filename='tsne-300.png'):
    assert low_dim_embs.shape[0] == len(labels), "Number of embeddings and labels should be equal"
    plt.figure(figsize=(12, 12))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


def cmp(x, y):
    if x < y:
        return x, y
    else:
        return y, x


if __name__ == '__main__':
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 300
    dir_load = './data'
    print('Sort the words by frequency')
    with open(dir_load+'/word_vector.pkl', 'rb') as f:
        word_dict = pickle.load(f)  # {word: {word:, freq:, possibility:, vector:, Huffman:}}
    # word_pos_dict = {k: v['possibility'] for k, v in word_dict.keys()}
    words_order = sorted(word_dict, key=lambda x: word_dict[x]['possibility'], reverse=True)
    embedding_word = words_order[: plot_only]
    with open(dir_load+'/word_vector_norm.pkl', 'rb') as f:
        word_vector_norm = pickle.load(f)
    embedding_vector = [word_vector_norm[w] for w in embedding_word]
    embedding_vector = np.array(embedding_vector).reshape([plot_only, -1])
    print(embedding_vector.shape)
    low_dim_embeddings = tsne.fit_transform(embedding_vector)
    plot_with_labels(low_dim_embeddings, embedding_word)
    print('DONE'.center(70, '='))

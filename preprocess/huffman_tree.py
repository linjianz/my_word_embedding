#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-19 21:31:45
Program: 
Description: 
"""
import numpy as np


class HuffmanTreeNode(object):
    """
    霍夫曼树结点
    """
    def __init__(self, value, possibility):
        self.value = value                  # 叶节点存储单词本身，非叶节点则会在merge的时候向量化
        self.possibility = possibility
        self.left = None
        self.right = None
        self.Huffman = ''                   # 霍夫曼编码

    def __str__(self):
        return 'HuffmanTreeNode object, value: {v}, possibility: {p}, Huffman code: {h}'\
            .format(v=self.value, p=self.possibility, h=self.Huffman)


class HuffmanTree(object):
    def __init__(self, word_dict, vec_len=150):
        self.word_dict = word_dict
        self.vec_len = vec_len
        self.root = None
        print('\n')
        print('Build Huffman Tree'.center(70, '='))
        print('Generate node')
        node_list = [HuffmanTreeNode(x['word'], x['possibility']) for x in word_dict.values()]
        print('Build tree')
        self.build_tree(node_list)
        print('Generate huffman code')
        self.generate_huffman_code()

    def build_tree(self, node_list):
        """
        循环：
            找到概率最小的两个结点，合并，把合并后的结点放入结点列表，合并前的两个结点弹出
        """
        print('process>>>')
        node_num = len(node_list)
        node_num_10 = node_num // 10
        while len(node_list) > 1:
            i1 = 0                          # 概率最小的结点
            i2 = 1                          # 概率第二小的结点
            if node_list[i1].possibility > node_list[i2].possibility:
                i1, i2 = i2, i1
            for i in range(2, len(node_list)):
                if node_list[i].possibility < node_list[i2].possibility:
                    i2 = i
                    if node_list[i].possibility < node_list[i1].possibility:
                        i1, i2 = i2, i1
            top_node = self.merge(node_list[i1], node_list[i2])
            if i1 > i2:                     # 必须先pop较大者
                node_list.pop(i1)
                node_list.pop(i2)
            elif i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0, top_node)
            dealt = node_num - len(node_list)
            if dealt % node_num_10 == 0:
                print('{}0%'.format(dealt//node_num_10))
        self.root = node_list[0]

    def generate_huffman_code(self):
        """
        非递归计算叶节点的编码
        """
        stack = [self.root]
        while stack.__len__() > 0:
            node = stack.pop()
            while node.left or node.right:
                code = node.Huffman
                node.left.Huffman = code + "1"
                node.right.Huffman = code + "0"
                stack.append(node.right)
                node = node.left
            word = node.value   # 这个node即为叶子结点
            code = node.Huffman
            self.word_dict[word]['Huffman'] = code

    def merge(self, node1, node2):
        top_possibility = node1.possibility + node2.possibility
        top_node = HuffmanTreeNode(np.zeros([1, self.vec_len]), top_possibility)
        if node1.possibility >= node2.possibility:  # 设置左子树概率更大（好像不需要？）
            top_node.left = node1
            top_node.right = node2
        else:
            top_node.left = node2
            top_node.right = node1
        return top_node

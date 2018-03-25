#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Linjian Zhang
Email: linjian93@foxmail.com
Create Time: 2018-03-20 15:00:29
Program: 
Description: 
"""
import numpy as np


def normalize(vector, norm='l2'):
    """
    vector: 1 x n
    """
    norms = 1
    if norm == 'l1':
        norms = np.abs(vector).sum(axis=1)
    elif norm == 'l2':
        norms = np.sqrt(np.multiply(vector, vector).sum(axis=1))
    elif norm == 'max':
        norms = np.max(vector, axis=1)
    if norms == 0:
        raise Exception('Norms error!')
    vector = np.divide(vector, norms)
    return vector


def sigmoid(value):
    return 1/(1+np.exp(-value))


if __name__ == '__main__':
    a = np.arange(4).reshape([1, 4])
    print(a)
    b = normalize(a).reshape([-1])
    print(b)
    print(b[1]*b[1]+b[2]*b[2]+b[3]*b[3])

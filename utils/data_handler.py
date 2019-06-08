#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.


"""
data handler script
"""

import numpy as np
import random
import math
import pandas as pd
def samplerFunction(x):
    return x + (x**2) + (x**4) + (x**5) + (x**7) + (x**8)
def genData(dim):
    x = np.random.uniform(0., 1., size=dim)
    y = np.sin(2*math.pi*x) + np.random.normal(scale=0.003, size=dim)

    return zip(x, y)


def getTrainTestSplit(data, ratio=0.8):
    random.shuffle(data)
    split_idx = int(len(data) * ratio)
    train = data[:split_idx]
    test = data[split_idx:]
    return train, test




def getFileData(data_file):
    df = pd.read_csv(data_file)
    data_list = df.values.tolist()
    data_XY= []
    for data in data_list:
        
        data_XY.append((data[:-1] + [1], data[-1]))
    return data_XY

def getTrainTestData(train_file, test_file):
    return getFileData(train_file), getFileData(test_file)



if __name__ == '__main__':
    print genData(5)
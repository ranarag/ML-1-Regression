#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.


"""
script for the forward pass of the regression model
"""

import math
def calc_mean(func):
    def inner(data, weights):
        if len(data) == 0:
            print "Cannot divide!, no input data"
            return

        return func(data, weights) /  len(data)
    return inner



def phi(X, weights):
    out = 0.
    # print X
    #  print len(weights)
    for i, (x, wt) in enumerate(zip(X, weights)):
        # print len(X)
        out += (wt * (x ** i))
    return out


@calc_mean
def squared_err(data, weights):
    err = 0.0
    for (x, y) in data:
        # print x
        # print y
        err += (phi(x, weights) - y) ** 2
    return err / 2.


@calc_mean
def abs_err(data, weights):
    err = 0.
    for x, y in data:
        err += abs(phi(x, weights) - y)
    return err




def sigmoid(X, weights):
    sigmoid =  1.0 / (1 + math.exp(-1 * phi(X, weights)))
    return sigmoid


@calc_mean
def logistic_regression_err(data, weights):
    err = 0.
    for (X, y) in data:
        h = sigmoid(X, weights)
        err += -((y * math.log(h))  + ((1 - y) * math.log(1- h)))
    return err


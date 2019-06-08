#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.


"""
script for the backward pass of the regression model
"""


from forward import phi

# decorator function to update gradient during backward pass
def update_gradient(func):
    def inner(data, weights, lr):
        grad_arr = func(data, weights, lr)
        new_wts = [0.0] * len(weights)
        for i in range(len(weights)):
            new_wts[i] = weights[i] - (lr * grad_arr[i])
        return new_wts
    return inner





# squared error function
@update_gradient
def grad_squared_err(data, weights, lr):
    X_dimension = len(data[0][0])
    grad_arr  = [0.] * X_dimension
    n = len(data)
    for (X, y) in data:
        for i, x in enumerate(X):
            grad_arr[i] += (phi(X, weights) - y) * x
    for i in range(X_dimension):
        grad_arr[i] /= n
    return grad_arr


# absolute error function
@update_gradient
def grad_abs_err(data, weights, lr):
    X_dimension = len(data[0][0])
    grad_arr  = [0.] * X_dimension
    n = len(data)

    for X, y in data:
        temp = (phi(X, weights) - y)
        for i, x in enumerate(X):
            grad_arr[i] += (temp / abs(temp)) * x
    for i in range(X_dimension):
        grad_arr[i] /= n
    return grad_arr

# logistic_error_function
@update_gradient
def grad_logistic_err(data, weights, lr):
    X_dimension = len(data[0][0])
    grad_arr  = [0.] * X_dimension
    n = len(data)

    for (X, y) in data:
        for i, x in enumerate(X):
            grad_arr[i] += (sigmoid(X, weights) - y) * x
    for i in range(X_dimension):
        grad_arr[i] /= n
    return grad_arr
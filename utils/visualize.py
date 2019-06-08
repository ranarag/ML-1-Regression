#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.


"""
plotting codes
"""

import matplotlib.pyplot as plt
from  models.forward import phi
import numpy as np

def createPlot(data, style, text):
    x = [val[0] for val in data]
    y = [val[1] for val in data]
    plt.plot(x, y, style, label = text)

def plotSaver(name, label, xlab = 'X', ylab = 'Y'):
	plt.xlabel(xlab)
	plt.ylabel(ylab)
	plt.legend()
	plt.title(label)
	plt.savefig(name)
	plt.clf()


def interpolatePlot(weights, numpts = 100):
	x = np.linspace(0, 1, numpts)
	data = [[val, phi(val, weights)] for val in x]
	return data
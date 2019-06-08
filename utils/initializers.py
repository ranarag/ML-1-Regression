#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.


"""
initilizer script
you can try with different scales of the normal distribution
"""

import numpy as np

def randomNormalInitializer(size):
    wts = np.random.normal(scale=0.5, size=size+1)
    return wts
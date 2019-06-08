#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.


"""
Solution to part 1 of the assignment
"""

# Imports
from utils.data_handler import *
from utils.initializers import *
from models.forward import *
from models.backward import *
import numpy as np
import cPickle as pkl
import json


# number of iterations each model is trained for
NUM_ITER = 3000
TRAIN_FILE='train.csv'
TEST_FILE = 'test.csv'


train, test = getTrainTestData(TRAIN_FILE, TEST_FILE)


print 'Train Test Data Loaded from files {} {}'.format(TRAIN_FILE, TEST_FILE)
parameter_dict = {}
parameter_dict[0] = {'train': train,
				     'test': test}


    
wts = randomNormalInitializer(1) # dimension of X is 1

for k in range(NUM_ITER):
	wts = grad_abs_err(train, wts, 0.0005) # returning absolute error as 
												 # well as updating it
	# error
	if k % 100 == 0:
		train_err = abs_err(train, wts)
		test_err = abs_err(test, wts)

		print "Train Error: {} : Test Error: {}".format(train_err, test_err)


# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:29:54 2016

@author: aggrey
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target

no_classes = len(np.unique(target))
'''
CREATING ONE-HOTENCODERS FROM CATEGORY DATA
'''
#METHOD ONE
result_one = np.zeros((target.shape[0], no_classes))
result_one[np.arange(target.shape[0]), target] = 1
print result_one

#METHOD TWO
result_two = pd.get_dummies(target).values
print result_two
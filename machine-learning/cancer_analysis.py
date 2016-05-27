# -*- coding: utf-8 -*-
"""
Created on Wed May  4 16:14:55 2016

@author: aggrey
"""

import numpy as np
import pandas as pd
from sklearn import metrics, cross_validation
from tensorflow.contrib import skflow
import os
import random

random.seed(42)

filename = filename = os.getcwd() + '/datasets/mesothelioma.csv'

dataset = pd.read_csv(filename, low_memory=False)
dataset = dataset.drop(['city'], axis=1)
dataset['class of diagnosis'] = np.where(dataset['class of diagnosis'] == 1, 0,1)
data = dataset.drop(['class of diagnosis'], axis=1)
target = dataset['class of diagnosis']
number_of_classes = len(np.unique(target))#2


xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(data, target,test_size=0.2, random_state=20)


classifier = skflow.TensorFlowDNNClassifier(hidden_units=[40,60, 40],
                                            n_classes=number_of_classes,
                                            steps=2000,
                                            learning_rate=.1)
classifier.fit(xtrain, ytrain)
score = metrics.accuracy_score(ytest, classifier.predict(xtest))
print('Accuracy: {0:f}'.format(score))
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 10:43:23 2016

@author: aggrey
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import skflow
import os
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from PIL import Image
from sklearn import metrics, cross_validation
def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def processImage(image_name):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    image = Image.open(image_name)
    image = image.convert('L')
    image = image.resize((28, 28))
    channels = 1
    data = np.array(image.getdata(),np.uint8).reshape(image.size[1], image.size[0], channels)
    data = np.asarray(data, dtype=np.int32)    
    return data
    





def loadIndoorScenes():
    x = []
    y = []
    dir_name = os.getcwd() + '/datasets/dsets/'
    dir_list = os.listdir(dir_name)
    for folder_id in range(len(dir_list)):
        folder = dir_name + dir_list[folder_id] + '/'
        sub_folder = os.listdir(folder)
        for j in range (len(sub_folder)):
            image_name = folder + sub_folder[j]
            x_input = processImage(image_name)
            x.append(x_input)
            y.append(folder_id)

    return x, y

images, labels = loadIndoorScenes()
dataset = np.asarray(images)
target = np.asarray(labels)

X = dataset


#num_of_labels = 10
#target = (np.arange(num_of_labels) == target[:, None]).astype(np.float32) # Convert to one-hot vector
y = target

def max_pool_2x2(tensor_in):
    return tf.nn.max_pool(tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')

def conv_model(X, y):
    X = tf.reshape(X, [-1, 28, 28, 1])
    with tf.variable_scope('conv_layer1'):
        h_conv1 = skflow.ops.conv2d(X, n_filters=32, filter_shape=[5, 5], bias=True, activation=tf.nn.relu)
        h_pool1 = max_pool_2x2(h_conv1)
    with tf.variable_scope('conv_layer2'):
        h_conv2 = skflow.ops.conv2d(h_pool1, n_filters=64, filter_shape=[5, 5], 
                                    bias=True, activation=tf.nn.relu)
        h_pool2 = max_pool_2x2(h_conv2)
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = skflow.ops.dnn(h_pool2_flat, [1024], activation=tf.nn.relu, dropout=0.5)
    return skflow.models.logistic_regression(h_fc1, y)


classifier = skflow.TensorFlowEstimator(model_fn=conv_model, n_classes=100, batch_size=100, steps=100000,learning_rate=0.05)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=10)
classifier.fit(X_train, y_train)
score = metrics.accuracy_score(y_test, classifier.predict(X_test))
print('Accuracy: {0:f}'.format(score))

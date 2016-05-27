# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:33:22 2016

@author: aggrey
"""

import os
from PIL import Image
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from sklearn import metrics, cross_validation
import tensorflow as tf


# Load an image and convert it to an numpy array
def processImage(image_name):
    image = Image.open(image_name)
    image = image.resize((28, 28), Image.ANTIALIAS)
    image = image.convert('L')
    image_array = np.asarray(image, dtype=np.float32)
    return image_array

#Open a folder containing the image, load the dataset,and create a list with image arrays
def prepareDataset():
    dataset = []
    dir_name = os.getcwd() + '/datasets/Folio/'
    dir_list = os.listdir(dir_name)
    for folder_id in range(len(dir_list)):
        folder = dir_name + dir_list[folder_id] + '/'
        sub_folder = os.listdir(folder)
        for j in range (len(sub_folder)):
            image_name = folder + sub_folder[j]
            image_array = processImage(image_name)
            dataset.append([image_array, folder_id])
    return dataset

vector_set = prepareDataset()
x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]

images = np.asarray(x_data)
labels = np.asarray(y_data)

#WORK ON IMAGES AND CONVERT THEN TO THE NECCESSARY TENSORS
#1. Convert to a 4D tensor with [number_of_images, rows, columns, color_channels]
num_images, rows, columns = images.shape
color_channels = 1
images = images.reshape(num_images, rows, columns, color_channels)
#2. convert shape from [number_of_images, rows, columns, depth] to
# to [num_of_images, rows*columns], assuming that the depth is still(color_of_channel) 1
assert images.shape[3] == 1
images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])


#convert from [0, 255] -> [0.0, 1.0] if data type is float32
if images.dtype == np.float32:
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0/255.0)

#WORK ON LABELS AND CONVERT THEM TO ONE EL-HOT
def dense_to_one_hot(labels_dense, num_classes):
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


num_classes = 10 #For now, we are using on 10 classes of leaves
labels = dense_to_one_hot(labels, num_classes)

# -------DONE PREPARING THE DATASET
# start of class
class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
        "images.shape: %s labels.shape: %s" % (images.shape,labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

# END OF CLASS
def read_data_sets(train_images, train_labels, test_images, test_labels):
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets
# End of function

x_train, x_test, y_train, y_test = cross_validation.train_test_split(images, labels,test_size=0.2)

mydata = read_data_sets(x_train, y_train, x_test, y_test)

x = tf.placeholder('float',[None, 784]) # Width(28) * height(28), None(number of rows in the dataset)
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10])) # We have 10 classes of output
matma = tf.matmul(x,W)
y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder('float', [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

session = tf.Session()
session.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mydata.train.next_batch(10)
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print session.run(accuracy, feed_dict={x: mydata.test.images, y_: mydata.test.labels})

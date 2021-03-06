# -*- coding: utf-8 -*-
"""
Created on Tue May 10 09:44:50 2016

@author: aggrey
"""

import os
from PIL import Image
import numpy as np
from sklearn import cross_validation
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt


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
            folder_name = dir_list[folder_id]
            dataset.append([image_array, folder_id,folder_name])
    return dataset

#vector_set = prepareDataset()
x_data = [v[0] for v in vector_set]
y_data = [v[1] for v in vector_set]
z_data = [v[2] for v in vector_set]

images = np.asarray(x_data)
labels = np.asarray(y_data)
leaves_names = np.asarray(z_data)

number_of_each_specie = np.bincount(labels)
leaves_labels =  np.unique(leaves_names)

def plot_distribution():
    x_pos = np.arange(len(number_of_each_specie))
    sns.set_context("poster")
    sns.set_style("ticks")
    #sns.despine()
    sns.barplot(x_pos, number_of_each_specie, palette="Set1", alpha=.8)
    sns.despine()
    plt.xticks(x_pos, leaves_labels, rotation = 90)
    plt.title('Number of leaves of each plant specie')
    plt.ylabel('Count')

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


num_classes = len(np.unique(labels)) #For now, we are using on 10 classes of leaves
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



# ----- here we go, dear convolutional neural networks------
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, num_classes])

x_image = tf.reshape(x, [-1,28,28,1])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #[?, 28, 28, 32]
h_pool1 = max_pool_2x2(h_conv1) #[?, 12,12, 32]

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) #[?,12, 12, 32, 64]
h_pool2 = max_pool_2x2(h_conv2)





W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, num_classes])
b_fc2 = bias_variable([num_classes])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()

sess.run(tf.initialize_all_variables())

for i in range(500000):
   batch = mydata.train.next_batch(10)
   if i%10 == 0:
     train_accuracy = sess.run( accuracy, feed_dict={ x:batch[0], y_: batch[1], keep_prob: 1.0})
     print("step %d, training accuracy %g"%(i, train_accuracy))
   sess.run(train_step,feed_dict={x: batch[0], y_: batch[1], keep_prob: .5})

print("test accuracy %g"% sess.run(accuracy, feed_dict={ 
       x: mydata.test.images, y_: mydata.test.labels, keep_prob: 1.0}))





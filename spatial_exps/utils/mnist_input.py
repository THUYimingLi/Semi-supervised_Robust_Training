"""
Utilities for importing the CIFAR10 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import sys
import tensorflow as tf
version = sys.version_info

import numpy as np
import idx2numpy

class MNISTData(object):
    """
    Unpickles the CIFAR10 dataset from a specified folder containing a pickled
    version following the format of Krizhevsky which can be found
    [here](https://www.cs.toronto.edu/~kriz/cifar.html).

    Inputs to constructor
    =====================

        - path: path to the pickled dataset. The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.

    """
    def __init__(self, path, partial=False, unlabeled_rate=1.):
        train_dataname = 'train-images-idx3-ubyte'
        train_labelname = 'train-labels-idx1-ubyte'
        test_dataname = 't10k-images-idx3-ubyte'
        test_labelname = 't10k-labels-idx1-ubyte'
        metadata_filename = 'batches.meta'

        train_images = np.zeros((50000, 28, 28, 1), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        mnist_images = idx2numpy.convert_from_file(os.path.join(path, train_dataname))
        mnist_labels = idx2numpy.convert_from_file(os.path.join(path, train_labelname))
        mnist_images = np.expand_dims(mnist_images, -1)
        train_images, train_labels = mnist_images, mnist_labels

        unlabeled_rate = min(1, unlabeled_rate)
        l = int(unlabeled_rate * 40000)
        unlabeled_images, unlabeled_labels = train_images[10000:10000+l], train_labels[10000:10000+l]
        
        if partial:
          train_images, train_labels = train_images[:10000], train_labels[:10000]
        eval_images, eval_labels = idx2numpy.convert_from_file(os.path.join(path, test_dataname)), idx2numpy.convert_from_file(os.path.join(path, test_labelname))
        eval_images = np.expand_dims(eval_images, -1)
        self.label_names = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

        self.train_data = Dataset(train_images, train_labels)
        self.eval_data = Dataset(eval_images, eval_labels)
        self.unlabeled_data = Dataset(unlabeled_images, unlabeled_labels)

    @staticmethod
    def _load_datafile(filename):
      with open(filename, 'rb') as fo:
          if version.major == 3:
              data_dict = pickle.load(fo, encoding='bytes')
          else:
              data_dict = pickle.load(fo)

          assert data_dict[b'data'].dtype == np.uint8
          image_data = data_dict[b'data']
          image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0,2,3,1)
          return image_data, np.array(data_dict[b'labels'])

class AugmentedMNISTData(object):
    """
    Data augmentation wrapper over a loaded dataset.

    Inputs to constructor
    =====================
        - raw_cifar10data: the loaded CIFAR10 dataset, via the CIFAR10Data class
        - sess: current tensorflow session
    """
    def __init__(self, raw_cifar10data, sess):
        assert isinstance(raw_cifar10data, MNISTData)
        self.image_size = 28

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32,
                                                  shape=[None, 28, 28, 1])

        # random transforamtion parameters
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img),
                            self.x_input_placeholder)

        self.augmented = flipped

        self.train_data = AugmentedDataset(raw_cifar10data.train_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.eval_data = AugmentedDataset(raw_cifar10data.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.unlabeled_data = AugmentedDataset(raw_cifar10data.unlabeled_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.label_names = raw_cifar10data.label_names


class Dataset(object):
    """
    Dataset object implementing a simple batching procedure.
    """
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False,
                       reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end],...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end],...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys


class AugmentedDataset(object):
    """
    Dataset object with built-in data augmentation. When performing 
    adversarial attacks, we cannot include data augmentation as part of the
    model. If we do the adversary will try to backprop through it. 
    """
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False,
                       reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size,
                                                       multiple_passes,
                                                       reshuffle_after_pass)
        images = raw_batch[0].astype(np.float32)
        return (self.sess.run(
                     self.augmented,
                     feed_dict={self.x_input_placeholder: raw_batch[0]}),
                raw_batch[1])

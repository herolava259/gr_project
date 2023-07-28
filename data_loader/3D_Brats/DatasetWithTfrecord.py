import tensorflow as tf
import numpy as np

from preprocessing.mri_image_feature_engineering import standardize

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'channel': tf.io.FixedLenFeature([],tf.int64),
    'label_raw': tf.io.FixedLenFeature([], tf.string),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}


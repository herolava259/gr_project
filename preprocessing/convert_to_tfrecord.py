import tensorflow as tf
import numpy as np
import nibabel as nib
import tarfile
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from preprocessing.mri_image_feature_engineering import get_sub_volume
import glob
import zipfile

N_LABEL = 4
def unzip_tar_file(zip_path, dest_dir):
    with tarfile(zip_path) as tar_zef:
        tar_zef.extractall(dest_dir)

def convert_nii_to_numpy(file_path):
    image_obj = nib.load(file_path)

    return image_obj.get_fdata()

def save_image_with_PIL(img_array, dest_file_path):
    img = Image.fromarray(img_array)
    img.save(dest_file_path)

def save_image_with_cv2(img_array, dest_file_path):
    cv2.imwrite(dest_file_path,img_array)

def show_img_with_cv2(img_array):
    cv2.imshow("Imgae",img_array)

def show_3d_img_with_plt(img_array):
    channel = np.random.randint(4, size=1)
    i = np.random.randint(0, 154)
    plt.imshow(img_array[:,:,i, channel])

nii_path = "../data/data_sample/img_raw/BRATS_001.nii.gz"

def show_test(img_arr):
    img_arr = convert_nii_to_numpy(nii_path)
    show_img_with_cv2(img_array= img_arr[:,:,1,1])



def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array

def image_example(image_array, label):

    mri_shape = image_array.shape

    feature = {
        'height': _int64_feature(mri_shape[0]),
        'width': _int64_feature(mri_shape[1]),
        'depth': _int64_feature(mri_shape[2]),
        'channel': _int64_feature(mri_shape[3]),
        'label_raw': _bytes_feature(serialize_array(label)),
        'image_raw': _bytes_feature(serialize_array(image_array))
    }

    return tf.train.Example(features=tf.train.Features(feature = feature))



def write_to_tfrecord_file(img_arr, label_arr, writer):

    tf_example = image_example(img_arr, label_arr)
    writer.write(tf_example.SerializeToString())


def process_raw_train_file_to_tf_record(img_dir, label_dir, record_file):
    img_file_list = glob.glob(img_dir + 'BRATS_[0-9]*.nii.gz')
    label_file_list = glob.glob(label_dir + 'BRATS_[0-9]*.nii.gz')

    sorted(img_file_list)
    sorted(label_file_list)
    print(len(img_file_list))
    print(len(label_file_list))

    with tf.io.TFRecordWriter(record_file) as writer:
        for im_f, lb_f in zip(img_file_list, label_file_list):
            print(im_f)
            print(lb_f)
            img_arr = convert_nii_to_numpy(im_f)
            label_arr = convert_nii_to_numpy(lb_f)

            print(img_arr.shape)
            img_arr, label_arr = get_sub_volume(img_arr, label_arr, first_channel=False)

            if (img_arr is None) or (label_arr is None):
                continue

            print('wrtite file to convert')
            write_to_tfrecord_file(img_arr, label_arr, writer)

image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'channel': tf.io.FixedLenFeature([],tf.int64),
    'label_raw': tf.io.FixedLenFeature([], tf.string),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

def wtest_convert_tfrecord(img_dir, label_dir, record_file):

    process_raw_train_file_to_tf_record(img_dir, label_dir, record_file)

    print("test_again")
    # Test converted tfrecord file
    raw_image_dataset = tf.data.TFRecordDataset(record_file)
    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:
        h,w,d,c = image_features['height'], image_features['width'], image_features['depth'], image_features['channel']
        img = tf.io.parse_tensor(image_features['image_raw'], out_type=tf.float64)
        img = tf.reshape(img, shape=[h,w,d,c]).numpy()
        print(f"shape of img of racord : ",img.shape)


def zip_directory(directory_path, zip_path):
    if not os.path.exists(directory_path):

        return

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)

                zipf.write(file_path, os.path.relpath(file_path, directory_path))
# if __name__ == 'main':
#     img_dir = '../data/data_sample/img_raw/'
#     label_raw = '../data/data_sample/label_raw/'
#
#     record_file = 'data1.tfrecords'
#
#     wtest_convert_tfrecord(img_dir, label_raw, record_file)

img_dir = '../data/data_sample/img_raw/'
label_raw = '../data/data_sample/label_raw/'

record_file = 'data2.tfrecords'

wtest_convert_tfrecord(img_dir, label_raw, record_file)

# raw_image_dataset = tf.data.TFRecordDataset(record_file)
# parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
# reader = tf.TFRecordReader()
# filename_queue = tf.train.string_input_producer(
#     record_file)
# _, serialized_example = reader.read(filename_queue)
# feature_set = {'image': tf.FixedLenFeature([], tf.string),
#                'label': tf.FixedLenFeature([], tf.int64)
#                }
#
# features = tf.parse_single_example(serialized_example, features=feature_set)
# label = features['label']
# print(f"shape of img of racord {id}: ", features['image_raw'].numpy().shape)
# for image_features in parsed_image_dataset:
#
#     print(f'type of img of record {id}: ', image_features['image_raw'])
#     print(f"shape of img of racord {id}: ", image_features['image_raw'].numpy().shape)

import tensorflow as tf
import numpy as np
import h5py
class VolumeDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 sample_list,
                 base_dir,
                 batch_size=1,
                 shuffle=True,
                 dim = (160, 160, 16),
                 num_channels=4,
                 num_classes=3,
                 verbose=1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.verbose = verbose
        self.sample_list = sample_list
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sample_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.sample_list) / self.batch_size))

    def __data_generation(self, list_IDs_temp):

        X = np.zeros((self.batch_size, self.num_channels, *self.dim),
                     dtype = np.float64)
        y = np.zeros((self.batch_size, self.num_classes, *self.dim),
                     dtype = np.float64)

        for i, ID in enumerate(list_IDs_temp):

            if self.verbose == 1:
                print("Training on: %s" % self.base_dir + ID)
            with h5py.File(self.base_dir + ID, 'r') as f:
                X[i] = np.array(f.get("x"))
                y[i] = np.array(f.get("y"))

        return X, y

    def __getitem__(self, index):

        indexes = self.indexes[
            index * self.batch_size: (index + 1) * self.batch_size
        ]

        sample_list_temp = [self.sample_list[k] for k in indexes]

        X, y = self.__data_generation(sample_list_temp)

        return X, y




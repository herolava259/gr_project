from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

from models.UnetFamily.AttentionUnet2D import attention_2D_model
from sklearn.model_selection import train_test_split
from pretraining.pretrainingAttnUnet2D import training_image_and_mask_data_generator, test_image_and_mask_data_generator
import os
import pandas as pd
from metrics.training.DiceLoss import dice_coef2D
from metrics.training.IouLoss import iou2D
import keras

EPOCHS = 5
BATCH_SIZE = 32
ImgHeight = 256
ImgWidth = 256
Channels = 3

#prepare training data

DataPath = "../data/2D_data/kaggle_3m"
dirs = []
images = []
masks = []
for dirname, _, filenames in os.walk(DataPath):
    for filename in filenames:
        if 'mask'in filename:
            dirs.append(dirname.replace(DataPath, ''))
            masks.append(filename)
            images.append(filename.replace('_mask', ''))

imagePath_df = pd.DataFrame({'directory':dirs, 'images':images,'masks':masks})

imagePath_df['image-path'] = DataPath + imagePath_df['directory'] + '/' + imagePath_df['images']
imagePath_df['mask-path'] = DataPath + imagePath_df['directory'] + '/' + imagePath_df['masks']

train, test = train_test_split(imagePath_df,test_size=0.25, random_state=21)

def data_iterator(image_gen,mask_gen):
    for img, mask in zip(image_gen,mask_gen):
        yield img, mask

timage_generator, tmask_generator = training_image_and_mask_data_generator(train, BATCH_SIZE,ImgHeight, ImgWidth)
vimage_generator,vmask_generator = test_image_and_mask_data_generator(test, BATCH_SIZE, ImgHeight, ImgWidth)

train_gen = data_iterator(timage_generator, tmask_generator)
val_gen = data_iterator(vimage_generator,vmask_generator)

# load 2D model

model = attention_2D_model(ImgHeight, ImgWidth)

model.compile(optimizer=Adam(), loss="binary_crossentropy",
             metrics=["accuracy", iou2D, dice_coef2D,keras.metrics.Precision()])
callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-5, verbose=1),
    ModelCheckpoint('attUNET-brain-mriv5.h5', verbose=1, save_best_only=True)
]

STEP_SIZE_TRAIN = timage_generator.n/BATCH_SIZE
STEP_SIZE_VALID = vimage_generator.n/BATCH_SIZE

results = model.fit(train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    batch_size=BATCH_SIZE,
                    epochs=10,
                    callbacks=callbacks,
                    validation_data=val_gen,
                   validation_steps=STEP_SIZE_VALID)

from keras.preprocessing.image import ImageDataGenerator

data_augmentation = dict(rotation_range = 0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest')


imagegen = ImageDataGenerator(rescale=1./255., **data_augmentation)
maskgen = ImageDataGenerator(rescale=1./255., **data_augmentation)


def training_image_and_mask_data_generator(train, BATCH_SIZE, ImgHeight, ImgWidth):
    image_generator= imagegen.flow_from_dataframe(dataframe=train,
                                 x_col="image-path",
                                 batch_size=BATCH_SIZE,
                                 seed=42,
                                 class_mode=None,
                                 target_size=(ImgHeight, ImgWidth),
                                 color_mode='rgb')
    mask_generator = maskgen.flow_from_dataframe(dataframe=train,
                                            x_col="mask-path",
                                            batch_size=BATCH_SIZE,
                                            seed=42,
                                            class_mode=None,
                                            target_size=(ImgHeight,ImgWidth),
                                            color_mode='grayscale')

    return image_generator, mask_generator


def test_image_and_mask_data_generator(test, BATCH_SIZE, ImgHeight, ImgWidth):
    imagegen = ImageDataGenerator(rescale=1. / 255.)
    maskgen = ImageDataGenerator(rescale=1. / 255.)

    # train generator
    vimage_generator = imagegen.flow_from_dataframe(dataframe=test,
                                                    x_col="image-path",
                                                    batch_size=BATCH_SIZE,
                                                    seed=42,
                                                    class_mode=None,
                                                    target_size=(ImgHeight, ImgWidth),
                                                    color_mode='rgb')
    # validation data generator
    vmask_generator = maskgen.flow_from_dataframe(dataframe=test,
                                                  x_col="mask-path",
                                                  batch_size=BATCH_SIZE,
                                                  seed=42,
                                                  class_mode=None,
                                                  target_size=(ImgHeight, ImgWidth),
                                                  color_mode='grayscale')

    return vimage_generator, vmask_generator


import tensorflow as tf
import keras
from keras.layers import (
    Activation,
    Conv3D,
    Conv3DTranspose,
    MaxPooling3D,
    UpSampling3D,
    Dropout,
    Concatenate
)

import keras.backend as K
from keras.layers import Layer, BatchNormalization

from keras.layers import Input
from keras.layers import concatenate
from keras import Model
from keras.optimizers import Adam
from metrics.loss.unet3dNormal import soft_dice_loss, dice_coefficient
K.set_image_data_format("channels_first")
def create_convolution_block(input_layer, n_filters, batch_normalization=False,
                             kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1),
                             instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(
        input_layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2),
                       strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)


def unet_model_3d(loss_function, input_shape=(16, 160, 160, 4),
                  pool_size=(2, 2, 2), n_labels=3,
                  initial_learning_rate=0.00001,
                  deconvolution=False, depth=4, n_base_filters=32,
                  include_label_wise_dice_coefficients=False, metrics=[],
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer,
                                          n_filters=n_base_filters * (
                                                  2 ** layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1,
                                          n_filters=n_base_filters * (
                                                  2 ** layer_depth) * 2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size,
                                            deconvolution=deconvolution,
                                            n_filters=
                                            current_layer.shape[1])(
            current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1].shape[1],
            input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1].shape[1],
            input_layer=current_layer,
            batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function,
                  metrics=metrics)
    return model






def create_3d_down_block(input_layers,
                 base_filters=32,
                 depth = 0,
                 kernel_size=(3,3,3),
                 activation=None, padding='same',
                 strides=(1,1,1),
                 pool_size = (2,2,2),
                 drop_out = 0.5,
                 has_batch_norm = False,
                 has_last_down = False,
                 has_drop_out = True):


    n_filters = base_filters * (2 ** depth)

    act_name = activation if activation is not None else 'relu'
    x = Conv3D(filters=n_filters, kernel_size=kernel_size,
               padding=padding,
               strides=strides,
               activation=act_name,
               name=f'Conv3D_Down_{depth + 1}_1')(input_layers)

    if has_batch_norm:
        x = BatchNormalization(trainable=True)(x)

    conv3d = Conv3D(filters=n_filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    strides=strides,
                    activation=act_name,
                    name=f'Conv3D_Down_{depth + 1}_2')(x)


    if has_batch_norm:
        x = BatchNormalization(trainable=True)(conv3d)
    else:
        x = conv3d

    if has_drop_out:
        x = Dropout(drop_out)(x)
    if has_last_down == False:
        x = MaxPooling3D(pool_size=pool_size)(x)

    return x, conv3d


def create_3d_up_block(
                 input_layer,
                 contracting_layer,
                 base_filters = 32,
                 depth = 0,
                 kernel_size=(2,2,2),
                 strides=(1,1,1),
                 padding = 'same',
                 activation= 'relu',
                 pool_size = (2,2,2),
                 deconvolution=False):

    act_name = activation if activation is not None else 'relu'
    up_conv_3d = UpSampling3D(size=pool_size,
                                   name=f'UpSampling3D_Up_{depth + 1}')(input_layer)
    if deconvolution:
        up_conv_3d = Conv3DTranspose(n_filters=base_filters * (2 ** depth), kernel_size=kernel_size,
                                          strides=strides,
                                          activation=act_name,
                                          name=f'Conv3DTranspose_Up_{depth + 1}'
                                          )

    concat = Concatenate(axis=1)([up_conv_3d,contracting_layer])
    conv3d = Conv3D(filters=base_filters * (2 ** (depth - 1)),
                         kernel_size=kernel_size,
                         padding=padding,
                         strides=strides,
                         activation=act_name,
                         name=f'Conv3D_Up_{depth + 1}')(concat)
    return conv3d

depth = 5
input_shape = (4,160,160,16)
def create_u_net_model_simple(loss_function,
                              initital_learning_rate = 0.00001,
                              input_shape =(4, 160, 160, 16), depth = 5,
                              has_batch_norm = False,
                              n_class= 3,
                              classifify_name = 'sigmoid',
                              metrics = []):
    input_layer = tf.keras.Input(input_shape)

    contract_layers = []

    x = input_layer
    for i in range(depth):

        if i < depth - 1:
            down_i, contract = create_3d_down_block(x, depth=i, has_batch_norm=has_batch_norm)
            contract_layers.append(contract)
        else:
            down_i, contract = create_3d_down_block(x, depth=i, has_batch_norm=has_batch_norm, has_last_down=True)
        x = down_i

    for i in range(depth - 2, -1, -1):
        up_i = None
        x = create_3d_up_block(x, contract_layers[i], depth=i)

    x = Conv3D(n_class,(1,1,1))(x)
    x = Activation(classifify_name)(x)

    model =  tf.keras.Model(inputs = input_layer, outputs = x)

    if not isinstance(metrics, list):
        metrics = [metrics]
    model.compile(optimizer=Adam(lr = initital_learning_rate), loss = loss_function,
                  metrics = metrics)
    return model



model = create_u_net_model_simple(loss_function=soft_dice_loss, metrics=[dice_coefficient],has_batch_norm=True)

model.summary()


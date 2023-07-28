import tensorflow as tf
import keras.backend as K
import keras

def diceLoss(y_true, y_pred, class_weights):

    y_true = tf.convert_to_tensor(y_true, "floatt32")
    y_pred = tf.convert_to_tensor(y_pred, y_true.dtype)

    num = tf.math.reduce_sum(
        tf.math.multiply(
            class_weights,
            tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0,1,2,3]),
        )

    )

    den = (
        tf.math.reduce_sum(
            tf.math.multiply(
                class_weights,
                tf.math.reduce_sum(tf.math.add(y_true, y_pred), axis = [0,1,2,3]),
            )
        )

        + 1e-5
    )

    return 1 - (2 * num / den)


# Keras
def DiceLoss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice

def dice_coef2D(ytrue, ypred):
    smoothing_factor=0.1
    ytrue_f = K.flatten(ytrue)
    ypred_f = K.flatten(ypred)
    intersection = K.sum(ytrue*ypred)
    ytrue_area = K.sum(ytrue)
    ypred_area = K.sum(ypred)
    combined_area = ytrue_area + ypred_area
    dice = 2*((intersection+smoothing_factor)/(combined_area+smoothing_factor))
    return dice

def dice_coef_loss2D(y_true, y_pred):
    return -dice_coef2D(y_true, y_pred)


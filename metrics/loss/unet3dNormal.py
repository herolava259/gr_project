import tensorflow as tf
import keras.backend as K


def single_class_dice_coefficient(y_true, y_pred, axis=(0,1,2),
                                  epsilon = 0.00001):
    dice_numerator = 2 * K.sum(y_true * y_pred) + epsilon
    dice_denominator = K.sum(y_true) + K.sum(y_pred) + epsilon
    dice_cofficient = dice_numerator / dice_denominator

    return dice_cofficient

def dice_coefficient(y_true, y_pred, axis=(1,2,3),
                    epsilon=0.00001):

    dice_numerator = 2*K.sum(y_pred*y_true, axis=axis) + epsilon
    dice_denominator = K.sum(y_pred, axis=axis) + K.sum(y_true, axis=axis) + epsilon
    dice_coefficient = K.mean(dice_numerator / dice_denominator)

    return dice_coefficient

def soft_dice_loss(y_true, y_pred, axis=(1,2,3),
                   epsilon=0.00001):

    dice_numerator = 2 * K.sum(y_pred*y_true, axis=axis) + epsilon
    dice_denominator = K.sum(y_pred**2, axis=axis) + K.sum(y_true**2, axis = axis) + epsilon

    dice_loss = 1 - K.mean(dice_numerator / dice_denominator)

    return dice_loss


import keras.backend as K
from metrics.training.IouLoss import  iou2D

def jac_distance2D(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    return -iou2D(y_true, y_pred)
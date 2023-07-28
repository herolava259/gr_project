import tensorflow as tf
from metrics.training.DiceLoss import DiceLoss

def generator_loss(target, gen_output, disc_fake_output, class_weights, alpha):

    # generalized dice loss
    dice_loss = DiceLoss(target, gen_output, class_weights)

    # disc loss
    disc_loss = tf.math.reduce_mean(
        tf.math.pow(tf.ones_like(disc_fake_output) - disc_fake_output, 2)
    )

    # total loss
    gen_loss = alpha * dice_loss + disc_loss

    dice_percent = (1 - dice_loss) * 100
    return gen_loss, dice_loss, disc_loss, dice_percent
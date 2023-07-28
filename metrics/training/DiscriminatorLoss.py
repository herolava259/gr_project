import tensorflow as tf


def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = tf.math.reduce_mean(
        tf.math.pow(tf.ones_like(disc_real_output) - disc_real_output, 2)
    )
    fake_loss = tf.math.reduce_mean(
        tf.math.pow(tf.zeros_like(disc_fake_output) - disc_fake_output, 2)
    )

    disc_loss = 0.5 * (real_loss + fake_loss)

    return disc_loss
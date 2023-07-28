import tensorflow as tf

def per_class_dice(y_true, y_pred, class_weights):
    y_true = tf.convert_to_tensor(y_true, "float32")
    y_pred = tf.convert_to_tensor(y_pred, y_true.dtype)
    weights = class_weights.copy()
    # WT, TC, ET
    per_class_losses = []
    for i in range(3):
        weights[i] = 0

        num = tf.math.reduce_sum(
            tf.math.multiply(
                weights,
                tf.math.reduce_sum(tf.math.multiply(y_true, y_pred), axis=[0, 1, 2, 3]),
            )
        )
        den = (
            tf.math.reduce_sum(
                tf.math.multiply(
                    weights,
                    tf.math.reduce_sum(tf.math.add(y_true, y_pred), axis=[0, 1, 2, 3]),
                )
            )
            + 1e-5
        )

        per_class_losses.append(2 * num / den)

    return per_class_losses
from tf.config import config

import tensorflow as tf

def get_criterion():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    return loss_object


def get_metrics():
    valid_loss = tf.keras.metrics.Mean(name="test_loss")

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")

    train_metrics = [train_accuracy]
    valid_metrics = [valid_loss, valid_accuracy]

    return train_metrics, valid_metrics

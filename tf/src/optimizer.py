from tf.config import config

import tensorflow as tf

def get_optimizer():
    return tf.keras.optimizers.Adam()


def get_scheduler():
    pass
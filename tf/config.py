import os
import yaml
from pprint import pprint

import tensorflow as tf


def connect_to_TPU(use_tpu: bool):
    """Detect hardware, return appropriate distribution strategy"""
    if use_tpu:
        try:
            # TPU detection. No parameters necessary if TPU_NAME environment variable is
            # set: this is always the case on Kaggle.
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print("Running on TPU ", tpu.master())
        except ValueError:
            tpu = None
    else:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    return tpu, strategy


def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


yaml.add_constructor("!join", join)

with open("config.yaml", "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # TPU
    tpu, strategy = connect_to_TPU(config['tpu'])
    print("World Size: ", strategy.num_replicas_in_sync)
    config['strategy'] = strategy
    config['autotune'] = tf.data.experimental.AUTOTUNE
    config['train_bs'] = config['train_bs'] * strategy.num_replicas_in_sync
    config['valid_bs'] = config['valid_bs'] * strategy.num_replicas_in_sync

    config['weights_save_prefix'] = os.path.join(config['weights_save'], 'ckpt')

    pprint("*** Configurations: ***")
    pprint(config)
    pprint("*** Config loaded succesfully. ***")
    pprint("----" * 10)

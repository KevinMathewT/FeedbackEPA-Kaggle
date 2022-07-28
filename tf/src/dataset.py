from tf.config import config

import numpy as np
import pandas as pd

import tensorflow as tf

from transformers import AutoTokenizer


def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_mask=True,
        return_token_type_ids=False,
        padding='max_length',
        max_length=maxlen,
        truncation=True,
    )

    return np.array(enc_di["input_ids"]), np.array(enc_di["attention_mask"])


def create_dist_dataset(ids, ams, y=None, training=False):
    dataset_ids = tf.data.Dataset.from_tensor_slices(ids)
    dataset_ams = tf.data.Dataset.from_tensor_slices(ams)
    dataset = tf.data.Dataset.zip(dataset_ids, dataset_ams)

    ### Add y if present ###
    if y is not None:
        dataset_y = tf.data.Dataset.from_tensor_slices(y)
        dataset = tf.data.Dataset.zip((dataset, dataset_y))

    ### Repeat if training ###
    if training:
        dataset = dataset.shuffle(len(ids)).repeat()

    if training:
        dataset = dataset.batch(config["train_bs"]).prefetch(config["autotune"])
    else:
        dataset = dataset.batch(config["valid_bs"]).prefetch(config["autotune"])

    ### make it distributed  ###
    if config['tpu']:
        dist_dataset = config["strategy"].experimental_distribute_dataset(dataset)

    return dist_dataset


def get_datasets(fold):
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], use_fast=True)

    df = pd.read_csv(config["train_folds"])
    train = df[df.fold != fold].reset_index(drop=True)
    valid = df[df.fold == fold].reset_index(drop=True)

    train_ids, train_ams = regular_encode(
        train.text.values.tolist(), tokenizer, maxlen=config["max_length"]
    )
    valid_ids, valid_ams = regular_encode(
        valid.text.values.tolist(), tokenizer, maxlen=config["max_length"]
    )

    train_y = train.target.values.reshape(-1, 1)
    valid_y = valid.target.values.reshape(-1, 1)

    train_dist_dataset = create_dist_dataset(train_ids, train_ams, train_y, True)
    valid_dist_dataset = create_dist_dataset(valid_ids, valid_ams, valid_y)

    return train_dist_dataset, valid_dist_dataset


if __name__ == "__main__":
    train_dist_dataset, valid_dist_dataset = get_datasets(0)

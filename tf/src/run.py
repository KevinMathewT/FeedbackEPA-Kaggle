from tf.config import config

import tensorflow as tf

from tf.src.dataset import get_datasets
from tf.src.loss import get_criterion, get_metrics
from tf.src.model import get_model
from tf.src.optimizer import get_optimizer
from tf.src.train import train

if __name__ == "__main__":
    fold = 0
    with config["strategy"].scope():
        model = get_model()
        train_dist_dataset, valid_dist_dataset = get_datasets(fold)
        optimizer = get_optimizer()
        criterion = get_criterion()
        train_metrics, valid_metrics = get_metrics()

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        train(
            model=model,
            train_dist_dataset=train_dist_dataset,
            valid_dist_dataset=valid_dist_dataset,
            optimizer=optimizer,
            criterion=criterion,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            checkpoint=checkpoint,
        )

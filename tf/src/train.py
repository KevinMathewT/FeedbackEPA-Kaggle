from tf.config import config

import tensorflow as tf


def train(
    model,
    train_dist_dataset,
    valid_dist_dataset,
    optimizer,
    criterion,
    train_metrics,
    valid_metrics,
    checkpoint,
):
    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = criterion(labels, predictions)
            loss = tf.nn.compute_average_loss(
                loss, global_batch_size=config["train_bs"]
            )

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        for metric in train_metrics:
            metric.update_state(labels, predictions)
        return loss

    def valid_step(inputs):
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = criterion(labels, predictions)

        valid_metrics[0].update_state(t_loss)
        for metric in valid_metrics[1:]:
            metric.update_state(labels, predictions)

    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = config["strategy"].run(train_step, args=(dataset_inputs,))
        return config["strategy"].reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    @tf.function
    def distributed_valid_step(dataset_inputs):
        return config["strategy"].run(valid_step, args=(dataset_inputs,))

    for epoch in range(config["epochs"]):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # valid LOOP
        for x in valid_dist_dataset:
            distributed_valid_step(x)

        if epoch % 2 == 0:
            checkpoint.save(config['weights_save_prefix'])

        template = (
            "Epoch {}, Loss: {}, Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}"
        )
        print(
            template.format(
                epoch + 1,
                train_loss,
                *[metric.result() for metric in train_metrics],
                *[metric.result() for metric in valid_metrics]
            )
        )

        for metric in train_metrics + valid_metrics:
            metric.reset_states()

from tf.config import config

import tensorflow as tf

import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Softmax
from keras.activations import tanh

from transformers import TFAutoModel, AutoConfig

# def simple_pooling_head():


class AttentionPooling(keras.layers.Layer):
    def __init__(self, model_config):
        super(AttentionPooling, self).__init__()
        self.drop = Dropout(rate=0.2)
        self.fc = Dense(config["num_classes"])
        self.attention = keras.Sequential(
            Dense(512, activation="tanh"), Dense(1, activation="softmax")
        )
        self.regressor = keras.Sequential(Dense(3))

    def call(self, model_out, attention_mask):
        last_hidden_state = model_out.last_hidden_state
        weights = self.attention(last_hidden_state)
        context_vector = tf.reduce_sum(weights * last_hidden_state, axis=1)
        outputs = self.regressor(context_vector)

        return outputs


class TFFeedBackModel(keras.Model):
    def __init__(self, **kwargs):
        super(TFFeedBackModel, self).__init__(**kwargs)
        self.config = AutoConfig.from_pretrained(config["model_name"])
        self.config.update({"output_hidden_states": True})
        self.model = TFAutoModel.from_pretrained(
            config["model_name"], config=self.config
        )

    def call(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask)
        out = self.pooler(out, mask)
        return out


def get_model():
    model = TFFeedBackModel()

    return model

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

import sys
import os


class FullyConnect(keras.layers.Layer):
    def __init__(
            self,
            units,
            kernel_initializer='random_normal',
            bias_initializer='zeros',
            use_bias=True
    ):
        super(FullyConnect, self).__init__()
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        return

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            trainable=True
        )

        if self.use_bias:
            self.b = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True
            )
        return

    def call(self, inputs, **kwargs):
        outputs = tf.matmul(inputs, self.W)

        if self.use_bias:
            outputs = tf.add(outputs, self.b)
        return outputs

    # def compute_mask(self, inputs, mask=None):
    #     return

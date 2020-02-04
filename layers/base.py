import tensorflow as tf
import tensorflow.keras as keras


class DeLayer(keras.layers.Layer):

    def __init__(self):
        keras.layers.Layer.__init__(self)

        return

    def build(self, input_shape):
        keras.layers.Layer.build(self, input_shape)

        return

    def call(self, inputs, **kwargs):
        keras.layers.Layer.call(self, inputs=inputs, **kwargs)

        return

    def compute_output_shape(self, input_shape):
        keras.layers.Layer.compute_output_shape(self, input_shape=input_shape)

        return

    def compute_mask(self, inputs, mask=None):
        keras.layers.Layer.compute_mask(self, inputs=inputs, mask=mask)

        return
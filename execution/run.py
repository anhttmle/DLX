from layers.dense import FullyConnect
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf


def run():
    print("START execution.run")
    fc = FullyConnect(5)
    print(fc(tf.Variable(shape=(10,10), initial_value=np.ones(shape=(10,10), dtype=np.float), dtype=tf.float32)))
    return


if __name__ == "__main__":
    run()

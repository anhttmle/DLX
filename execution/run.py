from layers.dense import FullyConnect
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

import os
import sys
import time

import pandas as pd


def run():
    print("START execution.run")

    data_dir = "/Users/DEMONEVIL/Documents/Credit Score"
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    train_data = []

    with open(train_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.rstrip("\n")
            cols = line.split(",")

            if index > 0:
                data = {}
                data["id"] = int(cols[0])
                data["label"] = int(cols[1])
                data["province"] = cols[2]
                data["district"] = cols[3]
                data["age_1"] = cols[4]
                data["age_2"] = cols[5]
                data["job"] = cols[6]

            print([cols[6]])
            if index > 10:
                return


    return train_df


if __name__ == "__main__":
    train_df = run()

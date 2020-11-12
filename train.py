"""
Original Author: Alex Cannan
Modifying Author: You!
Date Imported: 
Purpose: This file contains a script meant to train a model.
"""

import os
import sys
import logging

import sklearn
import sklearn.model_selection
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import model as _model
import utils

# dir setup
DATA_DIR = os.path.join(".", "data")
BIN_DIR = os.path.join(DATA_DIR, "bin")
OUTPUT_DIR = os.path.join(".", "output")


# TODO: Set hyperparameters
hyperparameter_grid: dict = {
    "optimizer": {
        "adam": tf.keras.optimizers.Adam,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "adagrad": tf.keras.optimizers.Adagrad
    },
    "loss": {"mse": keras.losses.MeanSquaredError,
     "mae": keras.losses.MeanAbsoluteError},
    "lr": [0.1, 0.01, 0.001, 0.0001],
    "batch_size": [8, 16, 32, 64, 128, 256],
    "dropout": [True, False],
    "batch_norm": [True, False],
    "epochs": [1, 3, 5, 7, 9, 11]
}

hyperparameters = {
    "optimizer": hyperparameter_grid["optimizer"]["adam"],
    "loss": hyperparameter_grid["loss"]["mse"],
    "lr": hyperparameter_grid["lr"][1],
    "batch_size": hyperparameter_grid["batch_size"][0],
    "dropout": hyperparameter_grid["dropout"][1],
    "batch_norm": hyperparameter_grid["batch_norm"][1],
    "epochs": hyperparameter_grid["epochs"][0],
}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    # TODO: Read training data and split into training and validation sets

    # I'm splitting into a training and validation set as I'm asked to,
    # but in reality I would have some process for cross validation which
    # I would do by invoking another call to the "train_test_split" function,
    # to create a validation set from the training data that we could use to validate
    # the results of training (otherwise we risk selecting hyperparameters which
    # overfit to the test set and not to the distribution of data seen in the training set)
    mos_list = utils.read_list(os.path.join(DATA_DIR, "mos_list.txt"))

    train_set, test_set = sklearn.model_selection.train_test_split(
        mos_list, test_size=0.2, shuffle=True,
    )

    logging.info(f"There are {len(train_set)} examples for training")

    # TODO: Initialize and compile model
    MOSNet = _model.CNN()
    model = MOSNet.build()
    model.compile(
        optimizer=hyperparameters["optimizer"](
            learning_rate = hyperparameters["lr"]
            ),
        loss=hyperparameters["loss"](),
    )
    # TODO: Start fitting model using utils.data_generator
    checkpoint_path = "./output/checkpoint.ckpt"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)
    history = model.fit(
        x=utils.data_generator(
            file_list=train_set,
            bin_root=BIN_DIR,
            batch_size=hyperparameters["batch_size"],
            frame=True,
        ),
        batch_size=hyperparameters["batch_size"],
        epochs=hyperparameters["epochs"],
        verbose=1,
        callbacks=[cp_callback],
    )

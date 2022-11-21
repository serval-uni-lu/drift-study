import typing as tp

import numpy as np
import tensorflow as tf
from mlc.models.tf_models import TfModel


class NnElectricity(TfModel):
    def __init__(self, name="nn_electricity", **kwargs):
        super().__init__(name=name, objective="binary", **kwargs)
        self.model = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: tp.Union[None, np.ndarray] = None,
        y_val: tp.Union[None, np.ndarray] = None,
    ):
        self.model = None
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                # tf.keras.layers.Dense(32, activation="relu"),
                # tf.keras.layers.Dropout(0.1),
                tf.keras.layers.Dense(1),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=["accuracy"],
        )
        self.model.fit(
            x, y, class_weight={0: y.mean(), 1: 1 - y.mean()}, epochs=10
        )


models = [("nn_electricity", NnElectricity)]

"""Constains callback to get performance metrics"""
import logging
import time
from statistics import mean

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class BenchmarkClbck(Callback):
    """Custom Keras Callback to compute training performance metrics"""

    def __init__(self, batch_size, nodes_number) -> None:
        self.batch_size = batch_size
        self.nodes_number = nodes_number
        self.global_samples_per_second = []
        super(BenchmarkClbck, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.__curr_epoch_time = time.time()
        self.steps_per_epochs = 0

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.__curr_epoch_time
        curr_global_samples_per_second = (
            self.steps_per_epochs * self.batch_size * self.nodes_number / epoch_time
        )
        self.global_samples_per_second.append(curr_global_samples_per_second)
        logging.info(
            "Epoch %d : %.2f global samples / second",
            epoch,
            curr_global_samples_per_second,
        )

    def on_train_batch_begin(self, batch, logs=None):
        self.steps_per_epochs += 1

    def on_train_end(self, logs=None):
        logging.info(
            "Average (on epochs) global samples/sec: %.2f",
            mean(self.global_samples_per_second),
        )

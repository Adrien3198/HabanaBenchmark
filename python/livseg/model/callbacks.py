"""Constains callback to get performance metrics"""
import logging
import time

import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class BenchmarkClbck(Callback):
    """Custom Keras Callback to compute training performance metrics"""

    def __init__(self, batch_size, nodes_number, logdir, logs_step=20) -> None:
        self.batch_size = batch_size
        self.nodes_number = nodes_number
        self.logs_step = logs_step
        self.total_time = 0
        self.total_steps = 0
        self.summary_writer = tf.summary.create_file_writer(logdir)

        super(BenchmarkClbck, self).__init__()

    @property
    def global_time(self):
        return self.total_time

    @property
    def global_steps(self):
        return self.total_steps * self.nodes_number

    @property
    def average_steps_per_second(self):
        return self.total_steps / self.total_time

    @property
    def global_average_steps_per_second(self):
        return self.global_steps / self.total_time

    def on_batch_begin(self, batch, logs):
        if self.total_steps % self.logs_step == 0:
            self.__curr_batch_time = time.time()

    def on_batch_end(self, batch, logs):
        self.total_steps += 1

        if self.total_steps % self.logs_step:
            return

        self.__curr_batch_time = time.time() - self.__curr_batch_time
        self.total_time += self.__curr_batch_time

        steps_per_seconds = self.logs_step / self.__curr_batch_time
        examples_per_second = steps_per_seconds * self.batch_size
        global_steps_per_seconds = steps_per_seconds * self.nodes_number
        global_examples_per_second = examples_per_second * self.nodes_number

        logging.info(
            "Step %d to %d : %.2f seconds, %.2f examples/sec",
            self.total_steps - self.logs_step,
            self.total_steps,
            self.__curr_batch_time,
            examples_per_second,
        )

        with self.summary_writer.as_default():
            tf.summary.scalar("examples/sec", examples_per_second, self.total_steps)
            tf.summary.scalar(
                "global_steps/sec", global_steps_per_seconds, self.global_steps
            )
            tf.summary.scalar(
                "global_examples/sec",
                global_examples_per_second,
                self.global_steps,
            )

    def on_train_end(self, logs=None):
        logging.info("Average steps/sec: %.2f", self.average_steps_per_second)
        logging.info(
            "Average examples/sec: %.2f",
            self.average_steps_per_second * self.batch_size,
        )
        logging.info("Total steps: %d", self.total_steps)
        logging.info("Total time: %d", self.total_time)
        logging.info("Global time: %.2f", self.global_time)
        logging.info("Global steps: %d", self.global_steps)
        logging.info("Global steps/sec: %.2f", self.global_average_steps_per_second)
        logging.info(
            "Global examples/sec: %.2f",
            self.global_average_steps_per_second * self.batch_size,
        )

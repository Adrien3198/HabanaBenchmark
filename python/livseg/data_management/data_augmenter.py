"""Contains data generator for keras and a dataset splitter"""

from numpy import ndarray
from pandas import DataFrame
import tensorflow as tf


class DataAugmenter:
    """Performs transformations for data augmentation"""

    def __init__(self, axes=(0, 1), seed=None) -> None:
        self.axes = axes
        self.horizontal_flip = tf.random.uniform(shape=(), seed=seed) > 0.5
        self.vertical_flip = tf.random.uniform(shape=(), seed=seed) > 0.5
        self.n_rots = tf.random.uniform(
            shape=(), dtype=tf.int32, minval=0, maxval=3, seed=seed
        )

    def flip(self, x: ndarray) -> ndarray:
        """Flips a 2D array"""
        x = tf.cond(
            pred=self.horizontal_flip,
            true_fn=lambda: tf.image.flip_left_right(x),
            false_fn=lambda: x,
        )
        x = tf.cond(
            pred=self.vertical_flip,
            true_fn=lambda: tf.image.flip_up_down(x),
            false_fn=lambda: x,
        )
        return x

    def rotate(self, x: ndarray) -> ndarray:
        """Rotate a 2D array"""
        return tf.image.rot90(x, k=self.n_rots)

    def transform(self, x: ndarray) -> ndarray:
        """Performs all transformations on a 2D array"""
        return self.flip(self.rotate(x))

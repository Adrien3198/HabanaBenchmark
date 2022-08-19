"""Contains data generator for keras and a dataset splitter"""

import enum
from os import PathLike
from typing import List, Tuple, Union

from livseg.data_management.loader import X_COL, Y_COL
from numpy import arange, fliplr, flipud, load, ndarray, random, rot90, stack
from pandas import DataFrame
from tensorflow.keras.utils import Sequence
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


class DatasetPartitioner:
    """Divide a dataset to be didtributed to multiple workers"""

    def __init__(self, df: DataFrame, num_workers: int) -> None:
        self.df = df.copy()
        self.num_workers = num_workers
        partition_size = len(df) // num_workers
        self.__splitted_indexes = [
            df.index[i * partition_size : (i + 1) * partition_size]
            for i in range(num_workers)
        ]

    def get_partition(self, partition_num: int) -> DataFrame:
        """Returns a partition of a Dataframe of paths"""
        return self.df.loc[self.__splitted_indexes[partition_num]]

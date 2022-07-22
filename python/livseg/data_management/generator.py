"""Contains data generator for keras and a dataset splitter"""

import enum
from os import PathLike
from typing import List, Tuple, Union

from livseg.data_management.loader import X_COL, Y_COL
from numpy import arange, fliplr, flipud, load, ndarray, random, rot90, stack
from pandas import DataFrame
from tensorflow.keras.utils import Sequence


class DataAugmenter:
    """Performs transformations for data augmentation"""

    class FlipType(enum.Enum):
        NONE = 0
        VERTICAL = 1
        HORIZONTAL = 2

    def __init__(self, flip_type: FlipType, rotation_number: int) -> None:
        self.flip_type = flip_type
        self.rotation_number = rotation_number

    def flip(self, x: ndarray) -> ndarray:
        """Flips a 2D array"""
        return [lambda x: x, fliplr, flipud][self.flip_type.value](x)

    def rotate(self, x: ndarray) -> ndarray:
        """Rotate a 2D array"""
        return rot90(x, self.rotation_number)

    def transform(self, x: ndarray) -> ndarray:
        """Performs all transformations on a 2D array"""
        return self.flip(self.rotate(x))


class DataGenerator(Sequence):
    """Generator of data batches for keras model fitting"""

    def __init__(
        self,
        df: DataFrame,
        batch_size=1,
        use_augmentation=False,
        shuffle=True,
    ) -> None:

        super().__init__()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.df = df.reset_index()
        self.indexes = arange(len(self.df))
        self.use_augmentation = use_augmentation
        self.on_epoch_end()

    @property
    def shapes(self):
        x, y = self[0]
        return x.shape, y.shape

    @property
    def dtypes(self):
        x, y = self[0]
        return x.dtype, y.dtype

    def __getitem__(self, index: int) -> Tuple[ndarray, ndarray]:
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        return self.__get_batch_data(batch_indexes)

    def __get_single_data(
        self, path_x: Union[str, PathLike], path_y: Union[str, PathLike]
    ) -> Tuple[ndarray, ndarray]:
        """Reads and processes a single pair image-label

        Parameters
        ----------
        path_x : Union[str, PathLike]
            path of the image
        path_y : Union[str, PathLike]
            path of the label mask

        Returns
        -------
        Tuple[ndarray, ndarray]
            pair of image-label
        """
        x = load(path_x).astype("float32")
        y = load(path_y).astype("int16")

        if not self.use_augmentation:
            return x, y

        flip_type = random.choice(DataAugmenter.FlipType)
        rotation_number = random.choice(range(4))
        data_augmenter = DataAugmenter(flip_type, rotation_number)

        return data_augmenter.transform(x), data_augmenter.transform(y)

    def __get_batch_data(
        self, batch_indexes: Union[ndarray, List]
    ) -> Tuple[ndarray, ndarray]:
        """Retrieves a batch of 2D slices with corresponding batch of labels from indexes

        Parameters
        ----------
        batch_indexes : Union[ndarray, List]
            arrays of batch indexes

        Returns
        -------
        Tuple[ndarray, ndarray]
            pair of images batch - labels batch
        """
        df = self.df.loc[batch_indexes]
        X, y = list(
            zip(
                *[
                    self.__get_single_data(*paths)
                    for paths in df[[X_COL, Y_COL]].values.tolist()
                ]
            )
        )
        return stack(X, axis=0), stack(y, axis=0)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.indexes)


class DatasetPartitioner:
    """Divide a dataset to be didtributed to multiple workers"""

    def __init__(self, df: DataFrame, num_workers: int) -> None:
        self.df = df.sample(frac=1)
        self.num_workers = num_workers
        partition_size = len(df) // num_workers
        self.__splitted_indexes = [
            df.index[i * partition_size : (i + 1) * partition_size]
            for i in range(num_workers)
        ]

    def get_partition(self, partition_num: int) -> DataFrame:
        """Returns a partition of a Dataframe of paths"""
        return self.df.loc[self.__splitted_indexes[partition_num]]

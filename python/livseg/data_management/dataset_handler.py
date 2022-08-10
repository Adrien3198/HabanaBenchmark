"""Contains dataset related class"""
import logging
import os
import re
import shutil
from typing import Iterable, Tuple, Union

import numpy as np
from livseg.data_management.loader import (
    X_COL,
    Y_COL,
    load_and_merge_ct_segmentations_paths,
    merge_ct_segmentations_paths,
    parse_filename,
    read_ct_array,
    read_segmentation_array,
)
from pandas import DataFrame
from tqdm import tqdm

TRAIN_NUMS = [
    123,
    39,
    70,
    47,
    59,
    120,
    105,
    91,
    84,
    29,
    62,
    65,
    86,
    18,
    19,
    1,
    50,
    24,
    81,
    64,
    41,
    97,
    56,
    122,
    69,
    12,
    63,
    119,
    6,
    21,
    93,
    73,
    58,
    8,
    15,
    75,
    46,
    57,
    36,
    26,
    14,
    17,
    94,
    43,
    60,
    79,
    67,
    9,
    33,
    128,
    0,
    74,
    99,
    32,
    125,
    28,
    72,
    22,
    38,
    103,
    53,
    118,
    31,
    114,
    85,
    129,
    107,
    10,
    109,
    3,
    115,
    37,
    121,
    4,
    90,
    2,
    124,
    117,
    30,
    16,
    83,
    89,
    104,
    110,
    49,
    77,
    80,
    130,
    55,
    13,
    45,
    106,
    66,
    44,
    25,
    40,
    11,
    113,
]
TEST_NUMS = [
    54,
    88,
    95,
    111,
    127,
    42,
    27,
    100,
    78,
    48,
    102,
    23,
    92,
    98,
    51,
    52,
    82,
    116,
    5,
    61,
    71,
    68,
    34,
    20,
    96,
    87,
    108,
    126,
    112,
    76,
    35,
    101,
    7,
]


class DatasetHandler:
    """Manage a directory where 2D slices of data are stored in train and test subdirectories"""

    pattern = re.compile(r"(?P<type>volume|segmentation)-(?P<num>\d+)-(?P<idx>\d+).npy")
    index_cols = ["num", "idx"]

    def __init__(self, directory: Union[str, os.PathLike]) -> None:
        self.directory = directory
        self.df = self.get_data_paths() if os.path.exists(directory) else None

    def get_data_paths(self) -> DataFrame:
        """Stores data paths in a Dataframe

        Returns
        -------
        DataFrame
            Dataframe of paths
        """
        df = DataFrame(
            [
                self.__get_infos_from_filename(filename)
                for filename in tqdm(os.listdir(self.directory))
            ]
        )
        return merge_ct_segmentations_paths(df, on=["num", "idx"])

    def get_train_test_split_from_ids(
        self, train_indexes: list = TRAIN_NUMS, test_indexes: list = TEST_NUMS
    ) -> Tuple[DataFrame, DataFrame]:
        """Splits Dataframe of data paths in train and test Dataframes of data from specified subjects numbers

        Parameters
        ----------
        train_indexes : list, optional
            List of subjects number in train set, by default TRAIN_NUMS
        test_indexes : list, optional
            List of subjects number in train set, by default TEST_NUMS

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            Dataframe of data paths of train and test sets
        """
        if self.df is None:
            raise ValueError("df attribute is not initialized")

        return (
            self.df[self.df["num"].isin(TRAIN_NUMS)].copy(),
            self.df[self.df["num"].isin(TEST_NUMS)].copy(),
        )

    def get_random_train_test_split(
        self, train_size=0.75
    ) -> Tuple[DataFrame, DataFrame]:
        """Splits randomly a Dataframe of data paths in train and test Dataframes of data

        Parameters
        ----------
        train_size : float, optional
            Part of the train set in all dataset, by default 0.75

        Returns
        -------
        Tuple[DataFrame, DataFrame]
            Dataframe of data paths of train and test sets
        """
        limit = int(len(self.df) * train_size)
        df = self.df.sample(frac=1, random_state=42)
        train_indexes, test_indexes = df.index[:limit], df.index[limit:]
        return df.loc[train_indexes].copy(), df.loc[test_indexes].copy()

    @staticmethod
    def create_directory(
        src_directory: Union[str, os.PathLike],
        tgt_directory: Union[str, os.PathLike],
        force: bool,
    ):
        """Creates a directory for training and testing sets of 2D slices of data

        Parameters
        ----------
        src_directory : Union[str, os.PathLike]
            Directory that contains preprocessed subjects data
        tgt_directory : Union[str, os.PathLike]
            Target directory where training and test data will be stored
        force : bool
            If True, overwrites data
        """
        if os.path.exists(tgt_directory):
            if force:
                shutil.rmtree(tgt_directory)
            else:
                return

        os.makedirs(tgt_directory)

        set_df = load_and_merge_ct_segmentations_paths(src_directory)

        for type in [X_COL, Y_COL]:
            print(f"Generate {type.split('_')[-1]} ...")
            DatasetHandler.__volumes_to_npys(tgt_directory, set_df[type])

    def __get_infos_from_filename(self, filename: str) -> dict:
        """Parses a file name to retrieve information about a slice of data

        Parameters
        ----------
        filename : str
            Name of file where a slice is stored

        Returns
        -------
        dict
            Informations about a slice of data
        """
        parsed = self.pattern.match(filename).groupdict()
        for k in self.index_cols:
            parsed[k] = int(parsed[k])
        parsed["path"] = os.path.join(self.directory, filename)
        return parsed

    @staticmethod
    def __single_volume_to_npys(
        directory: Union[str, os.PathLike], path: Union[str, os.PathLike]
    ):
        """Generates and saves 2D slices of subject volume

        Parameters
        ----------
        directory : Union[str, os.PathLike]
            Parent target directory of 2D slices
        path : Union[str, os.PathLike]
            Volume path of subject
        """
        parsed = parse_filename(path.split(os.path.sep)[-1])
        _type = parsed["type"]
        num = parsed["num"]

        if _type == "volume":
            array = read_ct_array(path)
        if _type == "segmentation":
            array = read_segmentation_array(path)

        array = np.expand_dims(array, axis=-1)

        for i, arr in enumerate(array):
            np.save(os.path.join(directory, f"{_type}-{num}-{i}.npy"), arr)

    @staticmethod
    def __volumes_to_npys(
        directory: Union[str, os.PathLike], paths: Iterable[Union[str, os.PathLike]]
    ):
        """Generates and saves 2D slices of multiple subjects volumes"""
        for path in tqdm(paths):
            DatasetHandler.__single_volume_to_npys(directory, path)

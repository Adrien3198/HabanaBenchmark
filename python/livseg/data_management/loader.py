"""Contains functions to read data"""

import os
import re
from typing import Dict, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk

pattern = re.compile(r"(?P<type>volume|segmentation)-(?P<num>\d+).nii")

__X_SUFFIX = "_ct"
__Y_SUFFIX = "_segmentation"

X_COL = f"path{__X_SUFFIX}"
Y_COL = f"path{__Y_SUFFIX}"


def parse_filename(filename: str) -> Dict[str, str]:
    """Parses a file name to retrieve information about a volume

    Parameters
    ----------
    filename : str
        File name of the volume

    Returns
    -------
    Dict[str, str]
        Information about a volume
    """
    try:
        parsed = pattern.match(filename).groupdict()
    except Exception as e:
        print(filename)
        raise e

    parsed["filename"] = filename
    return parsed


def reverse_parse_filename(type: str, num: int) -> str:
    """Create a file name from volume info"""
    return f"{type}-{num}.nii"


def load_data_paths_to_df(dataset_dir: Union[str, os.PathLike]) -> pd.DataFrame:
    """Create Dataframe of volume data paths

    Parameters
    ----------
    dataset_dir : Union[str, os.PathLike]
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    assert os.path.exists(
        dataset_dir
    ), f'Dataset directory "{dataset_dir}" does not exist'

    df = pd.DataFrame([parse_filename(fn) for fn in os.listdir(dataset_dir)])
    df["path"] = df["filename"].map(lambda x: os.path.join(dataset_dir, x))
    df["num"] = df["num"].astype(int)
    del df["filename"]
    return df


def merge_ct_segmentations_paths(df: pd.DataFrame) -> pd.DataFrame:
    """From Dataframe of volume data path, create a Dataframe with pairs of ct volume path and segmentation volume path

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of volume data path

    Returns
    -------
    pd.DataFrame
        Dataframe of ct-segmentation pairs of volume path
    """
    df_volumes = df[df["type"] == "volume"]
    df_segmentations = df[df["type"] == "segmentation"]

    return pd.merge(
        left=df_volumes,
        right=df_segmentations,
        on="num",
        suffixes=[__X_SUFFIX, __Y_SUFFIX],
    ).set_index("num")


def load_and_merge_ct_segmentations_paths(
    dataset_dir: Union[str, os.PathLike]
) -> pd.DataFrame:
    """Combines `load_data_paths_to_df` and `merge_ct_segmentations_paths` functions"""
    return merge_ct_segmentations_paths(load_data_paths_to_df(dataset_dir))


def read_ct(path: Union[str, os.PathLike]) -> sitk.Image:
    """Reads a CT volume in float32"""
    return sitk.ReadImage(path, sitk.sitkFloat32)


def read_segmentation(path: Union[str, os.PathLike]) -> sitk.Image:
    """Reads a CT volume in unint16"""
    return sitk.ReadImage(path, sitk.sitkUInt16)


def read_ct_array(path: Union[str, os.PathLike]) -> np.ndarray:
    """Reads a CT volume and returns an array"""
    return sitk.GetArrayFromImage(read_ct(path))


def read_segmentation_array(path: Union[str, os.PathLike]) -> np.ndarray:
    """Reads a label volume and returns an array"""
    return sitk.GetArrayFromImage(read_segmentation(path))


def save_image_to(image: sitk.Image, filename: str):
    """Saves a volume"""
    sitk.WriteImage(image, filename)

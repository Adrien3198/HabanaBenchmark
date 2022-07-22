"""Contains preprocessing functions"""

import os
from abc import abstractmethod
from typing import Tuple, Union

import numpy as np
import SimpleITK as sitk
from livseg.data_management.loader import (
    X_COL,
    Y_COL,
    read_ct,
    read_segmentation,
    save_image_to,
)
from pandas import DataFrame
from tqdm import tqdm


def array_to_image(ct_array: np.ndarray, direction, origin, spacing) -> sitk.Image:
    new_ct = sitk.GetImageFromArray(ct_array)
    new_ct.SetDirection(direction)
    new_ct.SetOrigin(origin)
    new_ct.SetSpacing(spacing)
    return new_ct


class Preprocessor:
    """Abstract class for data preprocessing"""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess_arr(self, array: np.ndarray) -> np.ndarray:
        pass

    def preprocess(self, image: sitk.Image) -> np.ndarray:
        volume_arr = sitk.GetArrayFromImage(image)
        volume_arr = self.preprocess_arr(volume_arr)
        return array_to_image(
            volume_arr, image.GetDirection(), image.GetOrigin(), image.GetSpacing()
        )


class CTPreprocessor(Preprocessor):
    """Preprocessing class for ct volumes"""

    def __init__(self, clipping_range=(-100, 250)) -> None:
        super(CTPreprocessor, self).__init__()
        self.clipping_range = clipping_range

    def hu_window(self, volume_arr: np.ndarray) -> np.ndarray:
        """Perform HU windowing to array values according a `self.clipping_range`

        Parameters
        ----------
        volume_arr : np.ndarray
            array to window

        Returns
        -------
        np.ndarray
            hu windowed array
        """
        result = np.copy(volume_arr)
        range_min, range_max = self.clipping_range
        result[volume_arr < range_min] = range_min
        result[volume_arr > range_max] = range_max
        return result

    def min_max_scaler(self, array: np.ndarray) -> np.ndarray:
        """Rescales voxels values to (-1, 1) according a `self.clipping_range`

        Parameters
        ----------
        array : np.ndarray
            array to rescale

        Returns
        -------
        np.ndarray
            rescaled array
        """
        min_val, max_val = self.clipping_range
        return (
            (2 * (array - min_val) / (max_val - min_val)) - 1
            if (max_val - min_val) != 0
            else np.copy(array)
        )

    def preprocess_arr(self, array: np.ndarray) -> np.ndarray:
        """Perform all preprocessing operation to an array"""
        x = self.hu_window(array)
        x = self.min_max_scaler(x)
        return x


class LabelPreprocessor(Preprocessor):
    """Preprocessing class for label volumes"""

    def __init__(self) -> None:
        super(LabelPreprocessor, self).__init__()

    def preprocess_arr(self, array: np.ndarray) -> np.ndarray:
        """Replace 2 to 1"""
        array[array > 0] = 1
        return array


__preprocessor_ct = CTPreprocessor(clipping_range=(-50, 250))
__preprocessor_label = LabelPreprocessor()


def preprocess(ct: sitk.Image, seg: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
    """Preprocesses ct-segmentation pair of volume

    Parameters
    ----------
    ct : sitk.Image
        ct image
    seg : sitk.Image
        segmentation image

    Returns
    -------
    Tuple[sitk.Image, sitk.Image]
        preprocessed ct-segmentation volume pair
    """
    new_ct = __preprocessor_ct.preprocess(ct)
    new_seg = __preprocessor_label.preprocess(seg)

    return new_ct, new_seg


def preprocess_from_df(df: DataFrame, target_dir: Union[str, os.PathLike]):
    """Preprocesses all volumes from paths Dataframe and saves in a new directory

    Parameters
    ----------
    df : DataFrame
        dataframe of data paths
    target_dir : Union[str, os.PathLike]
        directory where the preprocessed volumes are stored
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for _, p_x, p_y in tqdm(list(df[[X_COL, Y_COL]].itertuples())):
        ct = read_ct(p_x)
        seg = read_segmentation(p_y)

        new_ct, new_seg = preprocess(ct, seg)

        get_path = lambda p: os.path.join(target_dir, os.path.basename(p))

        save_image_to(new_ct, get_path(p_x))
        save_image_to(new_seg, get_path(p_y))

"""Contains the model"""
from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from .layers import *


def unet2D(input_shape: Tuple, first_filters=16, depth=4) -> Model:
    """Creates a 2D U-Net model"""

    inputs = Input(shape=input_shape)

    encoder_blocks = [DoubleConv2DBlock(first_filters)(inputs)]

    for i in range(1, depth):
        block = ConvDown2DBlock(first_filters * 2**i)(encoder_blocks[-1])
        encoder_blocks.append(block)

    decoder_blocks = [encoder_blocks[-1]]

    for i in reversed(range(depth - 1)):
        block = ConvUp2DBlock(first_filters * 2**i)(
            decoder_blocks[-1], encoder_blocks[i]
        )
        decoder_blocks.append(block)

    output = Conv2D(
        1, kernel_size=3, padding="same", activation="sigmoid", dtype="float32"
    )(decoder_blocks[-1])

    return Model(inputs, output)

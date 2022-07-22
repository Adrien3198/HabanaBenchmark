"""Contains functional layers for a 2D U-Net"""

from tensorflow.keras.initializers import glorot_uniform, zeros
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Layer,
)

__conv_args = dict(
    padding="same",
    kernel_initializer=glorot_uniform(16),
    bias_initializer=zeros(),
)


def __conv_down_2D(filters: int) -> Conv2D:
    """Downsampling convolution"""
    return Conv2D(filters, 3, strides=2, padding="same")


def __conv_up_2D(filters: int) -> Conv2DTranspose:
    """Upsampling convolution"""
    return Conv2DTranspose(filters, kernel_size=2, strides=2, **__conv_args)


def Conv2DBlock(filters: int):
    """Conv2D with BatchNormalization and ReLu activation layer

    Parameters
    ----------
    filters : int
        number of filters for the convolutional layer
    """

    def layer(x: Layer) -> Layer:
        x = Conv2D(filters=filters, kernel_size=3, **__conv_args)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    return layer


def DoubleConv2DBlock(filters: int):
    """Block of double 2D convolutions block"""

    def layer(x: Layer) -> Layer:
        x = Conv2DBlock(filters)(x)
        x = Conv2DBlock(filters)(x)
        return x

    return layer


def ConvDown2DBlock(filters: int):
    """Downsampling block (do)"""

    def layer(x: Layer) -> Layer:
        x = __conv_down_2D(filters)(x)
        x = DoubleConv2DBlock(filters)(x)
        return x

    return layer


def ConvUp2DBlock(filters: int):
    """Upsampling block and skip connections"""

    def layer(x: Layer, concat_layer: Layer) -> Layer:
        x = Concatenate()([__conv_up_2D(filters)(x), concat_layer])
        x = DoubleConv2DBlock(filters)(x)
        return x

    return layer

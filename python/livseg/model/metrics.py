"""Contains metrics to evaluate model"""
import tensorflow as tf


def __intersection_union(y_true, y_pred):
    """Compute the intersection and the union of label and predicted mask"""
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    return intersection, union


def dice_coef(y_true, y_pred):
    """Compute the Dice Coefficient between label and predicted mask"""
    eps = 1e-6
    intersection, union = __intersection_union(y_true, y_pred)
    return (2 * intersection + eps) / (union + eps)


def iou(y_true, y_pred):
    """Compute the Intersection Over Union metric between label and predicted mask"""
    eps = 1e-6
    intersection, union = __intersection_union(y_true, y_pred)
    return (intersection + eps) / (union - intersection + eps)

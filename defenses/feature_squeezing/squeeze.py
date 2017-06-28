import tensorflow as tf
import numpy as np
from scipy import ndimage

import cv2

def adaptive_binarize(x, block_size=5, C=33.8):
    "Works like an edge detector."
    # ADAPTIVE_THRESH_GAUSSIAN_C, ADAPTIVE_THRESH_MEAN_C
    # THRESH_BINARY, THRESH_BINARY_INV
    func = lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    return opencv_binarize(x, func)


def otsu_binarize(x):
    func = lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return opencv_binarize(x, func)


def opencv_binarize(x, func):
    x = np.rint(np.squeeze(x) * 255).astype(np.uint8)
    x_ret = []
    for img in x:
        img_bin = func(img)
        x_ret.append(img_bin)
    return np.expand_dims((np.array(x_ret)/255.).astype(np.float32), 3)


def reduce_precision_np(x, npp):
    """
    Reduce the precision of image, the numpy version.
    :param x: a float tensor, which has been scaled to [0, 1].
    :param npp: number of possible values per pixel. E.g. it's 256 for 8-bit gray-scale image, and 2 for binarized image.
    :return: a tensor representing image(s) with lower precision.
    """
    # Note: 0 is a possible value too.
    npp_int = npp - 1
    x_int = np.rint(x * npp_int)
    x_float = x_int / npp_int
    return x_float


def reduce_precision_tf(x, npp):
    """
    Reduce the precision of image, the tensorflow version.
    """
    npp_int = npp - 1
    x_int = tf.rint(tf.multiply(x, npp_int))
    x_float = tf.div(x_int, npp_int)
    return x_float


def binary_filter_tf(x):
    """
    An efficient implementation of reduce_precision_tf(x, 2).
    """
    x_bin = tf.nn.relu(tf.sign(x-0.5))
    return x_bin


def binary_filter_np(x):
    """
    An efficient implementation of reduce_precision_np(x, 2).
    """
    x_bin = np.maximum(np.sign(x-0.5), 0)
    return x_bin


def median_filter_np(x, width, height=-1):
    """
    Median smoothing by Scipy.
    :param x: a tensor of image(s)
    :param width: the width of the sliding window (number of pixels)
    :param height: the height of the window. The same as width by default.
    :return: a modified tensor with the same shape as x.
    """
    if height == -1:
        height = width
    return ndimage.filters.median_filter(x, size=(1,width,height,1), mode='reflect')


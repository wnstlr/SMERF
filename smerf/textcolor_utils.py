import numpy as np

import innvestigate
import innvestigate.utils as iutils
import innvestigate.utils.visualizations as ivis
import matplotlib.pyplot as plt 

def preprocess(X, net):
    X = X.copy()
    X = net["preprocess_f"](X)
    return X

def postprocess(X, color_conversion, channels_first):
    X = X.copy()
    X = iutils.postprocess_images(
        X, color_coding=color_conversion, channels_first=channels_first)
    return X

def image(X):
    X = X.copy()
    #return ivis.project(X, absmax=255.0, input_is_positive_only=True)
    return project(X, absmax=255.0, input_is_positive_only=True)


def identity(X):
    return X

def bk_proj(X):
    X = ivis.clip_quantile(X, 1)
    #return ivis.project(X)
    return project(X)

def heatmap(X):
    #X = ivis.gamma(X, minamp=0, gamma=0.95)
    #return ivis.heatmap(X)
    return heatmap(X)

def graymap(X):
    #return ivis.graymap(np.abs(X), input_is_positive_only=True)
    return _graymap(np.abs(X), input_is_positive_only=True)

def _graymap(X, **kwargs):
    """Same as :func:`heatmap` but uses a gray colormap."""
    return heatmap(X, cmap_type="gray", **kwargs)

def heatmap(X, cmap_type="seismic", reduce_op="sum", reduce_axis=-1, alpha_cmap=False, **kwargs):
    """Creates a heatmap/color map.
    Create a heatmap or colormap out of the input tensor.
    :param X: A image tensor with 4 axes.
    :param cmap_type: The color map to use. Default 'seismic'.
    :param reduce_op: Operation to reduce the color axis.
      Either 'sum' or 'absmax'.
    :param reduce_axis: Axis to reduce.
    :param alpha_cmap: Should the alpha component of the cmap be included.
    :param kwargs: Arguments passed on to :func:`project`
    :return: The tensor as color-map.
    """
    cmap = plt.cm.get_cmap(cmap_type)

    tmp = X
    shape = tmp.shape

    if reduce_op == "sum":
        tmp = tmp.sum(axis=reduce_axis)
    elif reduce_op == "absmax":
        pos_max = tmp.max(axis=reduce_axis)
        neg_max = (-tmp).max(axis=reduce_axis)
        abs_neg_max = -neg_max
        tmp = np.select([pos_max >= abs_neg_max, pos_max < abs_neg_max],
                        [pos_max, neg_max])
    else:
        raise NotImplementedError()

    tmp = project(tmp, output_range=(0, 255), **kwargs).astype(np.int64)

    if alpha_cmap:
        tmp = cmap(tmp.flatten()).T
    else:
        tmp = cmap(tmp.flatten())[:, :3].T
    tmp = tmp.T

    shape = list(shape)
    shape[reduce_axis] = 3 + alpha_cmap
    return tmp.reshape(shape).astype(np.float32)

def project(X, output_range=(0, 1), absmax=None, input_is_positive_only=False):
    """Projects a tensor into a value range.
    Projects the tensor values into the specified range.
    :param X: A tensor.
    :param output_range: The output value range.
    :param absmax: A tensor specifying the absmax used for normalizing.
      Default the absmax along the first axis.
    :param input_is_positive_only: Is the input value range only positive.
    :return: The tensor with the values project into output range.
    """

    if absmax is None:
        absmax = np.max(np.abs(X),
                        axis=tuple(range(1, len(X.shape))))
    absmax = np.asarray(absmax)

    mask = absmax != 0
    if mask.sum() > 0:
        if len(X.shape) == 3:
            X[mask] /= absmax[mask][:, np.newaxis, np.newaxis]
        elif len(X.shape) == 4:
            X[mask] /= absmax[mask][:, np.newaxis, np.newaxis, np.newaxis]

    if input_is_positive_only is False:
        X = (X+1)/2  # [0, 1]
    X = X.clip(0, 1)

    X = output_range[0] + (X * (output_range[1]-output_range[0]))
    return X

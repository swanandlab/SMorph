from os import path
import numpy as np

def imnorm(im):
    im = (im - im.min()) / (im.max() - im.min())
    return im


def only_name(fpath):
    name = '.'.join(path.basename(fpath).split('.')[:-1])
    return name


def _unwrap_polygon(polygon):
    if type(polygon) != np.ndarray:
        X, Y = polygon.xs, polygon.ys
    else:
        polygon = list(zip(*polygon))
        X, Y = polygon[0], polygon[1]
    return X, Y

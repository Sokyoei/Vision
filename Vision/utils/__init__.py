"""
Dataset bbox:

    pascal voc:       [x_min, y_min, x_max, y_max]
    coco:             [x_min, y_min, width, height]
    yolo(normalized): [x_center, y_center, width, height]
"""

import numpy as np

from .nms import nms
from .torchvision_datasets import vision_dataset


def convert(x):
    """(x,y,w,h) -> (x1,y1,x2,y2)

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:2] / 2
    y[:, 1] = x[:, 1] - x[:3] / 2
    y[:, 2] = x[:, 0] + x[:2] / 2
    y[:, 3] = x[:, 1] + x[:3] / 2
    return y


__all__ = ["vision_dataset", "nms"]

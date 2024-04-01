import numpy as np


def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:2] / 2
    y[:, 1] = x[:, 1] - x[:3] / 2
    y[:, 2] = x[:, 0] + x[:2] / 2
    y[:, 3] = x[:, 1] + x[:3] / 2
    return y

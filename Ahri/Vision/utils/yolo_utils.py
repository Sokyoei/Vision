"""
Dataset bbox:

    pascal voc:       [x_min, y_min, x_max, y_max]
    coco:             [x_min, y_min, width, height]
    yolo(normalized): [x_center, y_center, width, height]
"""

from typing import List

import cv2
import numpy as np
from numpy.typing import NDArray


def xywh_to_xyxy(x) -> NDArray:
    """(x,y,w,h) -> (x1,y1,x2,y2)

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def plot_image(yolo_results: NDArray, image: NDArray, CLASSES: List[str]):
    cv2.namedWindow("img", cv2.WINDOW_FREERATIO)

    for x1, y1, x2, y2, score, classes_index in yolo_results:
        x1, y1, x2, y2, classes_index = int(x1), int(y1), int(x2), int(y2), int(classes_index)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{CLASSES[classes_index]} {score:.2}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def nms(dets: NDArray, thresh: float):
    """
    NMS(Non-Max Supperssion) 非极大值抑制

    Args:
        dets (ndarray): [[x1,y1,x2,y2,score,class],...]
        thresh (float): thresh
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    keep = []
    index = scores.argsort()[::-1]

    while index.size > 0:
        i = index[0]
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)

        overlaps = w * h

        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]
    return keep

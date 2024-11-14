"""
几何变换
"""

import cv2
import numpy as np
from cv2.typing import MatLike

from Vision.utils.cv2_utils import PopstarAhri, img_show


@img_show("warpAffine")
def affine(img: MatLike) -> MatLike:
    """仿射变换"""
    height, width, _ = img.shape
    M = cv2.getAffineTransform(
        np.float32([[0, 0], [0, height], [width, 0]]), np.float32([[0, 0], [width / 2, height], [width, 0]])
    )
    dst = cv2.warpAffine(img, M, (width, height))
    return dst


@img_show("warpPerspective")
def perspective(img: MatLike) -> MatLike:
    """透视变换"""
    height, width, _ = img.shape
    M = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [0, height], [width, 0], [width, height]]),
        np.float32([[width / 2, 0], [0, height], [width, 0], [width / 2, height]]),
    )
    dst = cv2.warpPerspective(img, M, (width, height))
    return dst


def main():
    img = cv2.resize(PopstarAhri, (0, 0), None, 0.2, 0.2)
    affine(img)
    perspective(img)


if __name__ == "__main__":
    main()

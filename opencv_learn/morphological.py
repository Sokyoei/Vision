"""
形态学操作
"""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from opencv_learn.cv2_utils import img_show

ROOT = Path(".").resolve().parent


@img_show("erode", cv2.WINDOW_FREERATIO)
def erode(img: NDArray):
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    dst = cv2.erode(img, kernel)
    return dst


@img_show("dilate", cv2.WINDOW_FREERATIO)
def dilate(img: NDArray):
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    dst = cv2.dilate(img, kernel)
    return dst


def main():
    img_path = ROOT / "data/Ahri/Popstar Ahri.jpg"
    img = cv2.imread(str(img_path))
    erode(img)
    dilate(img)


if __name__ == '__main__':
    main()

"""
filter 滤波
"""

from pathlib import Path

import cv2
from cv2.typing import MatLike

from opencv_learn.cv2_utils import img_show

ROOT = Path(".").resolve().parent


@img_show("mean_blur", cv2.WINDOW_FREERATIO)
def mean_blur(img: MatLike):
    dst = cv2.blur(img, (3, 3))
    return dst


@img_show("median_blur", cv2.WINDOW_FREERATIO)
def median_blur(img: MatLike):
    dst = cv2.medianBlur(img, 3)
    return dst


@img_show("stack_blur", cv2.WINDOW_FREERATIO)
def stack_blur(img: MatLike):
    dst = cv2.stackBlur(img, (3, 3))
    return dst


@img_show("gaussian_blur", cv2.WINDOW_FREERATIO)
def gaussian_blur(img: MatLike):
    dst = cv2.GaussianBlur(img, (3, 3), 0.1)
    return dst


def main():
    img_path = ROOT / "data/Ahri/Popstar Ahri.jpg"
    img = cv2.imread(str(img_path))
    mean_blur(img)
    median_blur(img)
    stack_blur(img)
    gaussian_blur(img)


if __name__ == "__main__":
    main()

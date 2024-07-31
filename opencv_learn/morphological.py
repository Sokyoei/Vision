"""
形态学操作
"""

import cv2
import numpy as np
from cv2.typing import MatLike

from opencv_learn.cv2_utils import PopstarAhri, img_show


@img_show("erode")
def erode(img: MatLike) -> MatLike:
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    dst = cv2.erode(img, kernel)
    return dst


@img_show("dilate")
def dilate(img: MatLike) -> MatLike:
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    dst = cv2.dilate(img, kernel)
    return dst


@img_show("open")
def open_(img: MatLike) -> MatLike:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return dst


@img_show("close")
def close(img: MatLike) -> MatLike:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return dst


@img_show("gradient")
def gradient(img: MatLike) -> MatLike:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return dst


@img_show("tophat")
def tophat(img: MatLike) -> MatLike:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    return dst


@img_show("blackhat")
def blackhat(img: MatLike) -> MatLike:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dst = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return dst


def main():
    erode(PopstarAhri)
    dilate(PopstarAhri)
    open_(PopstarAhri)
    close(PopstarAhri)
    gradient(PopstarAhri)
    tophat(PopstarAhri)
    blackhat(PopstarAhri)


if __name__ == '__main__':
    main()

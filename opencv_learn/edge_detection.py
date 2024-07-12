"""
Edge Detection 边缘检测
"""

from pathlib import Path

import cv2
from numpy.typing import NDArray

from opencv_learn.cv2_utils import img_show

ROOT = Path(".").resolve().parent


@img_show("canny", cv2.WINDOW_FREERATIO)
def canny(img: NDArray):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(img, 100, 150)
    return dst


def main():
    img_path = ROOT / "data/Ahri/Popstar Ahri.jpg"
    img = cv2.imread(str(img_path))
    canny(img)


if __name__ == "__main__":
    main()

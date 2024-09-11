"""
opencv-python utils
"""

from functools import wraps
from typing import Tuple

import cv2

from Vision import VISION_ROOT

# colors for OpenCV
RED = [0, 0, 255]
GREEN = [0, 255, 0]
BLUE = [255, 0, 0]
BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
YELLOW = [0, 255, 255]
VIOLET = [238, 130, 238]
PINK = [203, 192, 255]
DEEPPINK = [147, 20, 255]
PURPLE = [128, 0, 128]
SKYBLUE = [230, 216, 173]
GOLD = [10, 215, 255]
DARKGRAY = [169, 169, 169]


def img_show(winname: str, flags=cv2.WINDOW_FREERATIO):
    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            dst = func(*args, **kwargs)
            cv2.namedWindow(winname, flags)
            cv2.imshow(winname, dst)
            cv2.waitKey()
            cv2.destroyAllWindows()

        return inner

    return wrapper


PopstarAhri = cv2.imread(str(VISION_ROOT / "data/Ahri/Popstar Ahri.jpg"))


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """opencv 16进制颜色字符串转换为 BGR 元组

    Args:
        hex_color (str): 16进制字符串

    Returns:
        Tuple[int, int, int]: BGR 元组
    """
    hex_color = hex_color.lstrip('#')
    assert len(hex_color) == 6
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

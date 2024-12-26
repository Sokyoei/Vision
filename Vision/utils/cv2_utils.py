"""
opencv-python utils
"""

from functools import wraps
from typing import Tuple

import cv2
import numpy as np
from cv2.typing import MatLike
from PIL import Image

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


def opencv_to_pillow(img: MatLike) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pillow_to_opencv(img: Image.Image) -> MatLike:
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def mouse_click(img: MatLike):
    """鼠标点击"""

    def mouse_callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            print(f"({x}, {y})")
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # 在图像上绘制一个红色圆点
            cv2.putText(img, f'({x}, {y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 在图像上显示坐标

    cv2.namedWindow("mouse_click", cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback("mouse_click", mouse_callback)
    while True:
        cv2.imshow("mouse_click", img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyWindow("mouse_click")

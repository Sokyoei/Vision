"""
opencv-python utils
"""

from functools import wraps
from typing import Literal, Tuple

import cv2
import numpy as np
from cv2.typing import MatLike
from PIL import Image

from Ahri.Vision import SOKYOEI_DATA_DIR

ColorType = Literal["red", "green", "blue", "yellow"]

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


def show_image(image: MatLike, winname: str, flags: int = cv2.WINDOW_FREERATIO) -> MatLike:
    """显示图像"""
    cv2.namedWindow(winname, flags)
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


def img_show(winname: str, flags: int = cv2.WINDOW_FREERATIO):
    """装饰器工厂函数，用于显示图像处理结果。

    该函数创建一个装饰器，该装饰器包装图像处理函数并在执行后显示结果图像.

    Args:
        winname (str): 显示窗口的名称
        flags (int, optional): 窗口属性标志. Defaults to cv2.WINDOW_FREERATIO.

    Returns:
        function: 装饰器函数
    """

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            dst = func(*args, **kwargs)
            show_image(dst, winname, flags)

        return inner

    return wrapper


def imread(path: str, flags=cv2.IMREAD_COLOR):
    """imread for CJK paths"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


PopstarAhri = cv2.imread(str(SOKYOEI_DATA_DIR / "Ahri/Popstar Ahri.jpg"))


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

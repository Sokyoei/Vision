"""
opencv-python utils
"""

from functools import wraps

import cv2

from Vision import VISION_ROOT


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

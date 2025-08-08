"""
模板匹配
"""

import cv2

from Ahri.Vision import VISION_ROOT
from Ahri.Vision.utils.cv2_utils import img_show


@img_show("match template")
def match_template():
    img = cv2.imread(str(VISION_ROOT / r"data\Ahri\Popstar Ahri.jpg"))
    templ = cv2.imread(r"D:\Antares\Ahri\Popstar Ahri\Popstar Ahri-1024x1024.png")
    tw, th, tz = templ.shape

    matchs = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matchs)
    cv2.rectangle(img, max_loc, (max_loc[0] + tw, max_loc[1] + th), (0, 255, 0), 2, cv2.LINE_AA)
    return img


def main():
    match_template()


if __name__ == "__main__":
    main()

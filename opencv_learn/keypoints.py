"""
keypoints 关键点检测
"""

from pathlib import Path
from typing import Literal

import cv2

from opencv_learn.cv2_utils import img_show

ROOT = Path(".").resolve().parent

KeyPointsType = Literal["sift", "orb"]


@img_show("keypoints", cv2.WINDOW_FREERATIO)
def keypoints(keypoints_type: KeyPointsType):
    img_path = ROOT / "data/Ahri/Popstar Ahri.jpg"
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    model = {"sift": cv2.SIFT_create, "orb": cv2.ORB_create}.get(keypoints_type)(20000)
    keypoints = model.detect(gray, None)
    cv2.drawKeypoints(img, keypoints, img, (0, 255, 0))
    return img


def main():
    keypoints("sift")
    keypoints("orb")


if __name__ == '__main__':
    main()

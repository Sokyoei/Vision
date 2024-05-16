from pathlib import Path

import cv2
import numpy as np

ROOT = Path(".").resolve().parent


def main():
    img_path = ROOT / "data/Ahri/Popstar Ahri.jpg"
    # img_path = ROOT / "data/Ahri/星之守护者 永绽盛芒 阿狸.jpg"
    cv2.namedWindow("Ahri", cv2.WINDOW_FREERATIO)
    src = cv2.imread(str(img_path))
    # HACK: CJK 路径，不建议这么做，尽量使用 ASCII 路径
    # src = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("Ahri", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

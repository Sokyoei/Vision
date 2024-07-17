from pathlib import Path

import cv2

ROOT = Path(".").resolve().parent


def main():
    img_path = ROOT / "data/Ahri/Popstar Ahri.jpg"
    # img_path = ROOT / "data/Ahri/星之守护者 永绽盛芒 阿狸.jpg"

    img = cv2.imread(str(img_path))
    # HACK: CJK 路径，不建议这么做，尽量使用 ASCII 路径
    # img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)

    # BGR -> BRG
    b, g, r = cv2.split(img)
    brg = cv2.merge([b, r, g])

    cv2.namedWindow("Ahri", cv2.WINDOW_FREERATIO)
    cv2.imshow("Ahri", brg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

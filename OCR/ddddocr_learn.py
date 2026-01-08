"""
https://github.com/sml2h3/ddddocr
"""

import cv2
import ddddocr

from Ahri.Asuka import ASUKA_ROOT
from Ahri.Asuka.utils.cv2_utils import opencv_to_pillow

IMG_PATH = ASUKA_ROOT / "images/9.jpg"


def main():
    ocr = ddddocr.DdddOcr()
    ocr.set_ranges(0)
    image = opencv_to_pillow(cv2.imread(IMG_PATH))
    result = ocr.classification(image, probability=False)
    print(result)
    # s = ""
    # for i in result['probability']:
    #     s += result['charsets'][i.index(max(i))]
    # print(s)


if __name__ == "__main__":
    main()

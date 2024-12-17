import pytesseract
from PIL import Image

from Vision import VISION_ROOT

IMG_PATH = VISION_ROOT / 'images/10.jpg'
TESSERACT_CONFIG = " ".join(
    [
        # "--psm",
        # "7",
        "-c",
        "tessedit_char_whitelist=0123456789",  # 设置白名单
        "tessedit_write_images=true",  # 设置输出日志
    ]
)


def main():
    image = Image.open(IMG_PATH)
    # using image_to_* API
    text = pytesseract.image_to_boxes(image, config=TESSERACT_CONFIG)
    print(text)


if __name__ == "__main__":
    main()

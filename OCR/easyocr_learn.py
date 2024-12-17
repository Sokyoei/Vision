import easyocr
from PIL import Image

from Vision import VISION_ROOT

IMG_PATH = VISION_ROOT / 'images/10.jpg'


def main():
    image = Image.open(IMG_PATH)

    reader = easyocr.Reader(["en"])
    results = reader.readtext(image, detail=0, allowlist="0123456789")
    print(results)


if __name__ == "__main__":
    main()

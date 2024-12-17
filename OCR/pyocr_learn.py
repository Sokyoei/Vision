import pyocr
import pyocr.builders
from PIL import Image

from Vision import VISION_ROOT

IMG_PATH = VISION_ROOT / 'images/README.jpg'


def main():
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        exit(1)
    ocr_tool = tools[0]
    image = Image.open(IMG_PATH)
    text = ocr_tool.image_to_string(image, lang='chi_sim', builder=pyocr.builders.TextBuilder())
    print(text)


if __name__ == "__main__":
    main()

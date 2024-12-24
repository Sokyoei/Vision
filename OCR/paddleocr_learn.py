import cv2
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

IMG_PATH = r"D:\Andromeda\Sokyoei\Vision\videos\running_start\00003.jpg"


def main():
    paddleocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    img = cv2.imread(IMG_PATH)
    result = paddleocr.ocr(img, cls=True)

    # for i in range(len(result[0])):
    #     print(result[0][i][1][0])

    for line in result:
        print(line)

    image = Image.open(IMG_PATH).convert('RGB')
    boxes = [elements[0] for elements in result[0]]
    txts = [elements[1][0] for elements in result[0]]
    scores = [elements[1][1] for elements in result[0]]

    im_show = draw_ocr(image, boxes, txts, scores)
    im_show = Image.fromarray(im_show)
    im_show.save('paddleocr_result.jpg')


if __name__ == "__main__":
    main()

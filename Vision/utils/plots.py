import cv2
from numpy.typing import NDArray


def plot_image(image: NDArray):
    cv2.namedWindow("img", cv2.WINDOW_FREERATIO)
    cv2.imshow("img", image)
    cv2.putText(image)

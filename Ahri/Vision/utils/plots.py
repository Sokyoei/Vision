from typing import List

import cv2
from numpy.typing import NDArray


def plot_image(yolo_results: NDArray, image: NDArray, CLASSES: List[str]):
    cv2.namedWindow("img", cv2.WINDOW_FREERATIO)

    for x1, y1, x2, y2, score, classes_index in yolo_results:
        x1, y1, x2, y2, classes_index = int(x1), int(y1), int(x2), int(y2), int(classes_index)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{CLASSES[classes_index]} {score:.2}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

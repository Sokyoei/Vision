import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results


def main():
    model = YOLO("yolov8n.pt")
    model.info()

    capture = cv2.VideoCapture(0)

    cv2.namedWindow("yolo", cv2.WINDOW_FREERATIO)

    # main loop
    while True:
        ret, img = capture.read()
        if not ret:
            break

        # do inference
        result: Results = model(img)

        cv2.imshow("yolo", result[0].plot())
        if cv2.waitKey(1) == 27:
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

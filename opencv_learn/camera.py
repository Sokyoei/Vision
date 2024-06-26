import cv2


def main():
    video = cv2.VideoCapture(0)
    fps = video.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        cv2.putText(frame, f"FPS: {fps}", [0, 15], cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1, cv2.LINE_AA)
        cv2.imshow("camera", frame)
        c = cv2.waitKey(1)

        if c == 27:  # ESC quit
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

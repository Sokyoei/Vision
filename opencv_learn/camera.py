import cv2

SCREEN_RECORDING = True  # 是否录屏
SAVE_PATH = "recording.mp4"  # 视频保存路径
CAMERA_INDEX = 0  # 摄像头下标


def main():
    capture = cv2.VideoCapture(CAMERA_INDEX)

    # camera props
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if SCREEN_RECORDING:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outer = cv2.VideoWriter(SAVE_PATH, fourcc, fps, (width, height), True)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if SCREEN_RECORDING:
            outer.write(frame)
        # cv2.putText(frame, f"FPS: {fps}", [0, 15], cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1, cv2.LINE_AA)
        cv2.imshow("camera", frame)
        c = cv2.waitKey(1)

        if c == 27:  # ESC quit
            break

    if SCREEN_RECORDING:
        outer.release()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

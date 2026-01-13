import cv2

from .base import AbstractCamera


class OpenCVCamera(AbstractCamera):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cap = cv2.VideoCapture()
        self.args = args
        self.kwargs = kwargs

    def run(self):
        self.cap.open(*self.args, **self.kwargs)
        while self.running:
            with self.frame_lock:
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
            with self.cond:
                self.cond.notify_all()
            cv2.waitKey(1)

    def __del__(self):
        self.cap.release()

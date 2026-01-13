import threading
from abc import ABC
from types import NoneType

from cv2.typing import MatLike


class AbstractCamera(threading.Thread, ABC):

    def __init__(self):
        super().__init__()

        # frame variable
        self.frame: MatLike | NoneType = None

        # thread variable
        self.running = True
        self.cond = threading.Condition()
        self.name = f"{self.__class__.__name__}Thread"
        self.frame_lock = threading.Lock()

"""
YOLOv5 ONNXRuntime

```shell
git clone https://github.com/ultralytics/yolov5
cd yolov5
conda create -n yolo python=3.9
pip install -r requirements.txt
python export.py --weights model_path --include=onnx
```
"""

from typing import List

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray

CLASSES = []


class YOLOv5ONNX(object):

    def __init__(self, onnx_path) -> None:
        model = onnx.load(onnx_path)

        try:
            onnx.checker.check_model(model)
        except Exception:
            print("model error")

        self.options = ort.SessionOptions()
        self.options.enable_profiling = True
        self.ort_session = ort.InferenceSession(onnx_path, self.options, ort.get_available_providers())
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

    def inference(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, [640, 640])
        # BGR -> RGB, HWC -> CHW
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        img = img.astype(np.float32)
        img /= 255.0
        img = np.expand_dims(img, axis=0)
        preds: List[NDArray] = self.ort_session.run([self.output_name], {self.input_name: img})
        return preds, img

    def postprocess(self, preds: List[NDArray]):
        """"""
        num_classes = preds[0].shape[1]
        for pred in preds:
            for each in pred.squeeze():
                print()


if __name__ == "__main__":
    yolov5 = YOLOv5ONNX("../models/yolov5s.onnx")
    preds, img = yolov5.inference("../images/bus.jpg")
    yolov5.postprocess(preds)
    img = img.squeeze().transpose(1, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("img", img)
    cv2.waitKey(0)

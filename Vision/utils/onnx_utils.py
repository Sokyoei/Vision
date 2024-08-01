from __future__ import annotations

import os
from abc import ABC, abstractmethod

import onnx
import onnxruntime as ort


class AbstractONNXLoader(ABC):

    def __init__(self, onnx_path: str | os.PathLike) -> None:
        """ONNX 模型加载器

        Args:
            onnx_path (str | os.PathLike): .onnx 模型路径

        Raises:
            f: 模型错误
        """
        super().__init__()
        self.onnx_path = onnx_path
        model = onnx.load(self.onnx_path)

        try:
            onnx.checker.check_model(model)
        except Exception:
            raise f"{onnx_path} model error"

        self.options = ort.SessionOptions()
        self.options.enable_profiling = True
        self.ort_session = ort.InferenceSession(self.onnx_path, self.options, providers=ort.get_available_providers())
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

    @abstractmethod
    def interface(self):
        """推理接口函数"""
        pass

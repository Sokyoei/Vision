"""
使用 onnx-graphsurgeon 创建 proto

Reference:
    https://github.com/kalfazed/tensorrt_starter/blob/main/chapter3-tensorrt-basics-and-onnx/3.5-onnxsurgeon/src/gs_create_conv.py
"""

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def main() -> None:
    input = gs.Variable(name="input0", dtype=np.float32, shape=(1, 3, 224, 224))
    weight = gs.Constant(name="conv1.weight", values=np.random.randn(5, 3, 3, 3))
    bias = gs.Constant(name="conv1.bias", values=np.random.randn(5))
    output = gs.Variable(name="output0", dtype=np.float32, shape=(1, 5, 224, 224))
    node = gs.Node(op="Conv", inputs=[input, weight, bias], outputs=[output], attrs={"pads": [1, 1, 1, 1]})
    graph = gs.Graph(nodes=[node], inputs=[input], outputs=[output])
    model = gs.export_onnx(graph)
    onnx.save(model, "sample-conv.onnx")


if __name__ == "__main__":
    main()

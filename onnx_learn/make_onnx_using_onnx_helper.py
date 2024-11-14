"""
使用 onnx.helper 创建 proto

Reference:
    https://github.com/kalfazed/tensorrt_starter/blob/main/chapter3-tensorrt-basics-and-onnx/3.3-read-and-parse-onnx/src/create_onnx_linear.py
"""

import onnx
from onnx import ModelProto, TensorProto, helper


def create_onnx() -> ModelProto:
    # 创建ValueProto
    a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
    b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [10, 10])

    # 创建NodeProto
    mul = helper.make_node('Mul', ['a', 'x'], 'c', "multiply")
    add = helper.make_node('Add', ['c', 'b'], 'y', "add")

    # 构建GraphProto
    graph = helper.make_graph([mul, add], 'sample-linear', [a, x, b], [y])

    # 构建ModelProto
    model = helper.make_model(graph)

    # 检查model是否有错误
    onnx.checker.check_model(model)

    # 保存model
    onnx.save(model, "sample-linear.onnx")

    return model


def main():
    model = create_onnx()
    print(model)


if __name__ == "__main__":
    main()

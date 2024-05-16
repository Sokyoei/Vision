# ONNX

[docs](https://onnx.ai/onnx/index.html)

> ONNX 模型是使用 Protobuf 序列化格式的文件

## onnx 模型格式

[onnx.proto](https://github.com/onnx/onnx/blob/main/onnx/onnx.proto)

```text
ModelProto
    |
    +-->GraphProto
            |
            +-->node(NodeProto) 模型计算节点
            +-->input(ValueInfoProto) 模型输入节点
            +-->output(ValueInfoProto) 模型输出节点
            +-->initializer(TensorProto) 模型权重参数
```

## ONNXRuntime

[docs](https://onnxruntime.ai/docs/)

> ONNX 模型部署

## 模型转换

```bash
pip install torch
pip install tf2onnx
pip install skl2onnx
```

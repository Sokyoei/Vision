# TensorRT

## 流程

- Logger    日志记录器
- Builder   用于创建 Network, 对模型序列化生成 engine
- Network   由 Builder 创建，最初是一个空容器
- Parser    用于解析 onnx 模型
- Context   上接 engine, 下接 inference

## 量化

- PTQ
- QAT

## .onnx 转换 .engine

```shell
trtexec --onnx=onnx_file --saveEngine=engine_file
```

## 参考

- [trt-samples-for-hackathon-cn](https://github.com/NVIDIA/trt-samples-for-hackathon-cn)
- [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx)

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>

// Logger 类，用于处理 TensorRT 日志信息
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << "TensorRT Log: " << msg << std::endl;
        }
    }
};

// 读取文件内容到缓冲区
std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        return buffer;
    }
    return {};
}

int main() {
    // 创建 Logger 实例
    Logger logger;

    // 创建 TensorRT 构建器
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    // 创建 ONNX 解析器
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    // 解析 ONNX 文件
    const std::string onnxFile = "sample-conv.onnx";
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX file: " << onnxFile << std::endl;
        return -1;
    }

    // 配置构建器
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20);  // 设置最大工作空间大小为 1MB

    // 构建 TensorRT 引擎
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine) {
        std::cerr << "Failed to build TensorRT engine." << std::endl;
        return -1;
    }

    // 创建推理上下文
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 获取输入和输出绑定索引
    int inputIndex = engine->getBindingIndex("input0");
    int outputIndex = engine->getBindingIndex("output0");

    // 获取输入和输出维度
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

    // 计算输入和输出数据大小
    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i) {
        inputSize *= inputDims.d[i];
    }
    size_t outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i) {
        outputSize *= outputDims.d[i];
    }

    // 分配 CPU 和 GPU 内存
    std::vector<float> inputHost(inputSize);
    std::vector<float> outputHost(outputSize);
    void* inputDevice;
    void* outputDevice;
    cudaMalloc(&inputDevice, inputSize * sizeof(float));
    cudaMalloc(&outputDevice, outputSize * sizeof(float));

    // 填充输入数据（示例）
    for (size_t i = 0; i < inputSize; ++i) {
        inputHost[i] = 1.0f;
    }

    // 将输入数据从 CPU 复制到 GPU
    cudaMemcpy(inputDevice, inputHost.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // 绑定输入和输出设备指针
    void* bindings[] = {inputDevice, outputDevice};

    // 执行推理
    context->enqueueV2(bindings, 0, nullptr);

    // 将输出数据从 GPU 复制到 CPU
    cudaMemcpy(outputHost.data(), outputDevice, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印输出结果（示例）
    for (size_t i = 0; i < outputSize; ++i) {
        std::cout << "Output[" << i << "]: " << outputHost[i] << std::endl;
    }

    // 释放资源
    cudaFree(inputDevice);
    cudaFree(outputDevice);
    context->destroy();
    engine->destroy();
    config->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();

    return 0;
}

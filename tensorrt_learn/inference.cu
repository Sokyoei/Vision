#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>

namespace Ahri {
// Logger 类，用于处理 TensorRT 日志信息
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << "TensorRT Log: " << msg << std::endl;
        }
    }
};

static inline Logger logger;

// 读取文件内容到缓冲区
std::vector<char> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (file.read(buffer.data(), size)) {
        return buffer;
    }
    return {};
}

int build_engine(std::string onnx_path, std::string engine_path) {
    // 创建 TensorRT 构建器
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));

    // 创建 ONNX 解析器
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    // 解析 ONNX 文件
    // const std::string onnxFile = "/media/supervisor/windowsd/Code/github/Vision/tensorrt_learn/sample-conv.onnx";
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX file: " << onnx_path << std::endl;
        return -1;
    }

    // 配置构建器
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    // config->setMaxWorkspaceSize(1 << 20);  // 设置最大工作空间大小为 1MB
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);

    // 构建 TensorRT 引擎
    auto engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!engine) {
        std::cerr << "Failed to build TensorRT engine." << std::endl;
        return -1;
    }
    // 序列化保存engine
    std::ofstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.is_open()) {
        std::cout << "Failed to open engine file" << std::endl;
        return -1;
    }

    engine_file.write((char*)engine->data(), engine->size());
    engine_file.close();

    std::cout << "Engine build success!" << std::endl;
    return 0;
}

int inference(std::string engine_path) {
    auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    auto engine_buffer = read_file(engine_path);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(engine_buffer.data(), engine_buffer.size()));
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

#if NV_TENSORRT_MAJOR > 8
    nvinfer1::Dims input_dim_shapes = engine->getTensorShape("input0");
    nvinfer1::Dims output_dim_shapes = engine->getTensorShape("output0");
#else
    // 获取输入和输出绑定索引
    int input_index = engine->getBindingIndex("input0");
    int output_index = engine->getBindingIndex("output0");

    // 获取输入和输出维度
    nvinfer1::Dims input_dim_shapes = engine->getBindingDimensions(input_index);
    nvinfer1::Dims output_dim_shapes = engine->getBindingDimensions(output_index);
#endif
    nvinfer1::Dims4 input_dims{1, input_dim_shapes.d[1], input_dim_shapes.d[2], input_dim_shapes.d[3]};
    nvinfer1::Dims4 output_dims{1, output_dim_shapes.d[1], output_dim_shapes.d[2], output_dim_shapes.d[3]};

    context->setInputShape("input0", input_dims);

    // 计算输入和输出数据大小
    size_t input_size = 1;
    for (int i = 0; i < input_dim_shapes.nbDims; ++i) {
        input_size *= input_dim_shapes.d[i];
    }
    size_t output_size = 1;
    for (int i = 0; i < output_dim_shapes.nbDims; ++i) {
        output_size *= output_dim_shapes.d[i];
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 分配 CPU 和 GPU 内存
    std::vector<float> input_host(input_size);
    std::vector<float> output_host(output_size);
    void* input_device;
    void* output_device;
    cudaMalloc(&input_device, input_size * sizeof(float));
    cudaMalloc(&output_device, output_size * sizeof(float));

    context->setInputTensorAddress("input0", input_device);
    context->setOutputTensorAddress("output0", output_device);

    // 填充输入数据（示例）
    for (size_t i = 0; i < input_size; ++i) {
        input_host[i] = 1.0f;
    }

    // 将输入数据从 CPU 复制到 GPU
    cudaMemcpyAsync(input_device, input_host.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 绑定输入和输出设备指针
    // void* bindings[] = {input_device, output_device};

    // 执行推理
    // context->enqueueV2(bindings, 0, nullptr);
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    // 将输出数据从 GPU 复制到 CPU
    cudaMemcpyAsync(output_host.data(), output_device, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

    // 打印输出结果（示例）
    for (size_t i = 0; i < output_size; ++i) {
        std::cout << "Output[" << i << "]: " << output_host[i] << std::endl;
    }

    // 释放资源
    cudaStreamDestroy(stream);
    cudaFree(input_device);
    cudaFree(output_device);
    return 0;
}
}  // namespace Ahri

int main() {
    std::string onnx_path{"/media/supervisor/windowsd/Code/github/Vision/tensorrt_learn/sample-conv.onnx"};
    std::string engine_path{"./sample-conv.engine"};
    Ahri::build_engine(onnx_path, engine_path);
    Ahri::inference(engine_path);
    return 0;
}

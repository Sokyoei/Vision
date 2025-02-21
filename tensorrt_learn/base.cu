#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <vector>
#include <cstring>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << "TensorRT Logger: " << msg << std::endl;
        }
    }
};

// 序列化引擎到文件
void serializeEngineToFile(nvinfer1::ICudaEngine* engine, const std::string& filename) {
    nvinfer1::IHostMemory* serializedEngine = engine->serialize();
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
        file.close();
    }
    serializedEngine->destroy();
}

// 从文件反序列化引擎
nvinfer1::ICudaEngine* deserializeEngineFromFile(const std::string& filename, Logger& logger, nvinfer1::IRuntime*& runtime) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.good()) {
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size, nullptr);
    return engine;
}

// 创建引擎
nvinfer1::ICudaEngine* createEngine(const std::string& onnxFile, Logger& logger) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO))) {
        std::cerr << "Failed to parse ONNX file." << std::endl;
        return nullptr;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 30); // 1GB

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    parser->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();

    return engine;
}

// 推理函数
void inference(nvinfer1::ICudaEngine* engine, const float* inputData) {
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // 输入输出缓冲区
    std::vector<void*> buffers(2);
    size_t inputSize = 1 * 3 * 224 * 224 * sizeof(float);
    size_t outputSize = 1 * 5 * 224 * 224 * sizeof(float);

    cudaMalloc(&buffers[0], inputSize);
    cudaMalloc(&buffers[1], outputSize);

    // 将输入数据从主机复制到设备
    cudaMemcpy(buffers[0], inputData, inputSize, cudaMemcpyHostToDevice);

    // 执行推理
    context->enqueueV2(buffers.data(), cudaStreamPerThread, nullptr);

    // 输出数据缓冲区
    std::vector<float> output(1 * 5 * 224 * 224);
    cudaMemcpy(output.data(), buffers[1], outputSize, cudaMemcpyDeviceToHost);

    // 打印推理结果
    std::cout << "Inference results:" << std::endl;
    for (size_t c = 0; c < 5; ++c) {
        std::cout << "Channel " << c << ":" << std::endl;
        for (size_t h = 0; h < 224; ++h) {
            for (size_t w = 0; w < 224; ++w) {
                size_t index = c * 224 * 224 + h * 224 + w;
                std::cout << output[index] << " ";
            }
            std::cout << std::endl;
        }
    }

    // 释放资源
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    context->destroy();
}

int main() {
    Logger logger;
    std::string onnxFile = "your_model.onnx";
    std::string engineFile = "your_engine.trt";

    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = deserializeEngineFromFile(engineFile, logger, runtime);
    if (!engine) {
        engine = createEngine(onnxFile, logger);
        if (engine) {
            serializeEngineToFile(engine, engineFile);
        }
    }

    if (!engine) {
        std::cerr << "Failed to create or load engine." << std::endl;
        if (runtime) {
            runtime->destroy();
        }
        return -1;
    }

    // 生成随机输入数据
    std::vector<float> input(1 * 3 * 224 * 224);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 执行推理
    inference(engine, input.data());

    engine->destroy();
    if (runtime) {
        runtime->destroy();
    }

    return 0;
}

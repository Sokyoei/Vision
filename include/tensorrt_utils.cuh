/**
 * @file tensorrt_utils.cuh
 * @date 2024/05/16
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef TENSORRT_UTILS_CUH
#define TENSORRT_UTILS_CUH

#include <fstream>
#include <iostream>
#include <string>

#ifdef __NVCC__
#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace Ahri {
namespace TRT {
/**
 * @brief tensorrt logger
 */
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} logger;

/**
 * @brief onnx convert to engine
 * @param onnx_path onnx model path
 * @param engine_path engine model path
 * @param build_flag @see nvinfer1::BuilderFlag
 */
void onnx_to_engine(std::string onnx_path, std::string engine_path, nvinfer1::BuilderFlag build_flag) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicit_batch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(onnx_path.c_str(), 2);
    for (int i = 0; i < parser->getNbErrors(); i++) {
        std::cout << "load error: " << parser->getError(i)->desc() << '\n';
    }
    std::cout << "TensorRT load mask onnx success" << '\n';

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(16 * (1 << 20));
    config->setFlag(build_flag);
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::ofstream file_ptr(engine_path, std::ios::binary);
    if (!file_ptr) {
        std::cerr << "could not open plan output file" << '\n';
        return;
    }
    nvinfer1::IHostMemory* model_stream = engine->serialize();
    file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());

    model_stream->destroy();
    engine->destroy();
    network->destroy();
    parser->destroy();
    std::cout << "convert onnx to tensorrt success" << '\n';
}

/**
 * @brief load tensorrt engine model
 * @param engine_path
 * @param engine
 * @param context
 */
void load_engine(const std::string& engine_path,
                 std::unique_ptr<nvinfer1::ICudaEngine>& engine,
                 std::unique_ptr<nvinfer1::IExecutionContext>& context) {
    std::ifstream file{engine_path, std::ios::binary};
    std::vector<char> data;

    file.seekg(0, file.end);
    const auto size = file.tellg();
    file.seekg(0, file.beg);

    data.resize(size);
    file.read(data.data(), size);
    file.close();

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
    context.reset(engine->createExecutionContext());
}
}  // namespace TRT
}  // namespace Ahri

#endif  // __NVCC__

#endif  // !TENSORRT_UTILS_CUH

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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#ifdef __NVCC__
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "Ahri.cuh"
#include "Ceceilia/utils/logger_utils.hpp"

namespace Ahri {
namespace TensorRT {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TensorRT Logger
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief tensorrt logger
 */
class TensorRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kVERBOSE:
                AHRI_LOGGER_DEBUG(msg);
            case Severity::kINFO:
                AHRI_LOGGER_INFO(msg);
            case Severity::kWARNING:
                AHRI_LOGGER_WARN(msg);
            case Severity::kERROR:
                AHRI_LOGGER_ERROR(msg);
            case Severity::kINTERNAL_ERROR:
                AHRI_LOGGER_CRITICAL(msg);
            default:
                AHRI_LOGGER_INFO(msg);
        }
    }
};

static inline TensorRTLogger trtlogger;

#define AHRI_TENSORRT_LOGGER_VERBOSE(...) trtlogger.log(Severity::kVERBOSE, fmt::format(__VA_ARGS__))
#define AHRI_TENSORRT_LOGGER_INFO(...) trtlogger.log(Severity::kINFO, fmt::format(__VA_ARGS__))
#define AHRI_TENSORRT_LOGGER_WARNING(...) trtlogger.log(Severity::kWARNING, fmt::format(__VA_ARGS__))
#define AHRI_TENSORRT_LOGGER_ERROR(...) trtlogger.log(Severity::kERROR, fmt::format(__VA_ARGS__))
#define AHRI_TENSORRT_LOGGER_INTERNAL_ERROR(...) trtlogger.log(Severity::kINTERNAL_ERROR, fmt::format(__VA_ARGS__))

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Inference
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief onnx convert to engine
 * @param onnx_path onnx model path
 * @param engine_path engine model path
 * @param build_flag @see nvinfer1::BuilderFlag
 */
void onnx_to_engine(std::string onnx_path, std::string engine_path, nvinfer1::BuilderFlag build_flag) {
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtlogger));
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trtlogger));
    parser->parseFromFile(onnx_path.c_str(), 2);
    for (int i = 0; i < parser->getNbErrors(); i++) {
        AHRI_LOGGER_ERROR("load error: {}", parser->getError(i)->desc());
    }
    AHRI_LOGGER_INFO("TensorRT load mask onnx success");

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    // config->setMaxWorkspaceSize(16 * (1 << 20));
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
    config->setFlag(build_flag);
    auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    std::ofstream file_ptr(engine_path, std::ios::binary);
    if (!file_ptr) {
        AHRI_LOGGER_ERROR("could not open plan output file");
        return;
    }
    std::unique_ptr<nvinfer1::IHostMemory> model_stream = std::unique_ptr<nvinfer1::IHostMemory>(engine->serialize());
    file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());

    AHRI_LOGGER_INFO("convert onnx to tensorrt success");
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

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(trtlogger));
    engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
    context.reset(engine->createExecutionContext());
}

class AbstractTensorRTInference {
public:
    /**
     * @brief Constructor
     * @param model_path onnx or engine model path
     */
    AbstractTensorRTInference(const std::filesystem::path model_path,
                              nvinfer1::BuilderFlag build_flag,
                              const bool auto_convert = true)
        : _model_path(model_path), _build_flag(build_flag) {
        // TODO
        _onnx_path = std::filesystem::path(_model_path);
        cudaStreamCreate(&_stream);
        load_engine();
    }

    ~AbstractTensorRTInference() { cudaStreamDestroy(_stream); }

    void inference() {
        // _context->enqueueV2(buffers, _stream, nullptr);
        cudaStreamSynchronize(_stream);
    }

    void load_engine() {
        std::ifstream file{_model_path, std::ios::binary};
        std::vector<char> data;

        file.seekg(0, file.end);
        const auto size = file.tellg();
        file.seekg(0, file.beg);

        data.resize(size);
        file.read(data.data(), size);
        file.close();

        _runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtlogger));
        _engine.reset(_runtime->deserializeCudaEngine(data.data(), data.size()));
        _context.reset(_engine->createExecutionContext());
    }

    void save_engine() {
        auto model_stream = std::unique_ptr<nvinfer1::IHostMemory>(_engine->serialize());
        std::ofstream f(_model_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
        f.close();
    }

    void onnx2engine() {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtlogger));
        const auto explicit_batch =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trtlogger));
        parser->parseFromFile(_onnx_path.c_str(), 2);
        for (int i = 0; i < parser->getNbErrors(); i++) {
            AHRI_LOGGER_ERROR("load error: {}", parser->getError(i)->desc());
        }
        AHRI_LOGGER_INFO("TensorRT load mask onnx success");

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        // config->setMaxWorkspaceSize(16 * (1 << 20));
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
        config->setFlag(_build_flag);
        auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
        std::ofstream file_ptr(_engine_path, std::ios::binary);
        if (!file_ptr) {
            AHRI_LOGGER_ERROR("could not open plan output file");
            return;
        }
        std::unique_ptr<nvinfer1::IHostMemory> model_stream =
            std::unique_ptr<nvinfer1::IHostMemory>(engine->serialize());
        file_ptr.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());

        AHRI_LOGGER_INFO("convert onnx to tensorrt success");
    }

private:
    std::filesystem::path _model_path;
    std::filesystem::path _onnx_path;
    std::filesystem::path _engine_path;
    nvinfer1::BuilderFlag _build_flag;
    cudaStream_t _stream;
    std::unique_ptr<nvinfer1::ICudaEngine> _engine;
    std::unique_ptr<nvinfer1::IExecutionContext> _context;
    std::unique_ptr<nvinfer1::IRuntime> _runtime;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Plugin
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define PLUGIN_GET_NAME_NAMESPACE_VERSION(name, namespace, version)           \
    nvinfer1::AsciiChar const* getPluginName() const noexcept override {      \
        return name;                                                          \
    }                                                                         \
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { \
        return namespace;                                                     \
    }                                                                         \
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {   \
        return version;                                                       \
    }
}  // namespace TensorRT
}  // namespace Ahri

namespace nvinfer1 {}  // namespace nvinfer1

#endif  // __NVCC__
#endif  // !TENSORRT_UTILS_CUH

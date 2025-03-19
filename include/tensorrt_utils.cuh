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
#include <vector>

#ifdef __NVCC__
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <fmt/core.h>
#include <fmt/std.h>

#include "Ahri.cuh"
#include "Ceceilia/utils/logger_utils.hpp"

namespace Ahri {
namespace TensorRT {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void print_network(nvinfer1::INetworkDefinition& network, nvinfer1::ICudaEngine& engine, bool optimized);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define ENGINE_EXTENSION ".engine"

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
                break;
            case Severity::kINFO:
                AHRI_LOGGER_INFO(msg);
                break;
            case Severity::kWARNING:
                AHRI_LOGGER_WARN(msg);
                break;
            case Severity::kERROR:
                AHRI_LOGGER_ERROR(msg);
                break;
            case Severity::kINTERNAL_ERROR:
                AHRI_LOGGER_CRITICAL(msg);
                break;
            default:
                AHRI_LOGGER_INFO(msg);
        }
    }
};

inline TensorRTLogger trtlogger;

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

class TensorRTModel {
public:
    TensorRTModel(std::filesystem::path onnx_path) : _onnx_path(onnx_path) {
        if (!std::filesystem::exists(_onnx_path)) {
            AHRI_LOGGER_ERROR("{} not found.", _onnx_path);
        }
        _engine_path = _onnx_path;
        _engine_path.replace_extension(ENGINE_EXTENSION);
    }
    ~TensorRTModel() {}

    bool build() {
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtlogger));
        constexpr auto explicit_batch =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trtlogger));
#ifdef _WIN32
        parser->parseFromFile(_onnx_path.wstring().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
#else
        parser->parseFromFile(_onnx_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
#endif
        for (int i = 0; i < parser->getNbErrors(); i++) {
            AHRI_LOGGER_ERROR("Parser ONNX error: {}", parser->getError(i)->desc());
        }
        AHRI_LOGGER_INFO("Parser ONNX success");

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
#if NV_TENSORRT_MAJOR >= 10
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
#else
        config->setMaxWorkspaceSize(1 << 28);
#endif

        auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
        auto model_stream = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

        // save to file
        std::ofstream engine_file(_engine_path, std::ios::binary);
        if (!engine_file) {
            AHRI_LOGGER_ERROR("Could not open {}", _engine_path);
            return false;
        }
        engine_file.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
        engine_file.close();

        // auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtlogger));
        // _engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        //     runtime->deserializeCudaEngine(model_stream->data(), model_stream->size()));
        // _input_dims = network->getInput(0)->getDimensions();
        // _output_dims = network->getOutput(0)->getDimensions();
        // AHRI_LOGGER_INFO("input dims: {}", _input_dims);
        // AHRI_LOGGER_INFO("output dims: {}", _output_dims);

        print_network(*network, *engine, false);
        print_network(*network, *engine, true);

        return true;
    }

    bool inference() {
        AHRI_LOGGER_INFO("Start inference");
        std::vector<uint8_t> model_data = load_file(_engine_path);

        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtlogger));
        _engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(model_data.data(), model_data.size()));
        auto context = std::unique_ptr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());

#if NV_TENSORRT_MAJOR >= 10
        nvinfer1::Dims input_dim_shapes = _engine->getTensorShape("input");
        nvinfer1::Dims output_dim_shapes = _engine->getTensorShape("output");
#else
        auto input_dims = context->getBindingDimensions(0);
        auto output_dims = context->getBindingDimensions(1);

        int input_index = _engine->getBindingIndex("input");
        int output_index = _engine->getBindingIndex("output");

        nvinfer1::Dims input_dim_shapes = _engine->getBindingDimensions(input_index);
        nvinfer1::Dims output_dim_shapes = _engine->getBindingDimensions(output_index);
#endif
        nvinfer1::Dims4 input_dims{1, input_dim_shapes.d[1], input_dim_shapes.d[2], input_dim_shapes.d[3]};
        nvinfer1::Dims4 output_dims{1, output_dim_shapes.d[1], output_dim_shapes.d[2], output_dim_shapes.d[3]};

        context->setInputShape("input", input_dims);

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

        // alloc cpu and gpu memory
        std::vector<float> input_host(input_size);
        std::vector<float> output_host(output_size);
        void* input_device;
        void* output_device;
        cudaMalloc(&input_device, input_size * sizeof(float));
        cudaMalloc(&output_device, output_size * sizeof(float));

        context->setInputTensorAddress("input", input_device);
        context->setOutputTensorAddress("output", output_device);

        for (size_t i = 0; i < input_size; ++i) {
            input_host[i] = 1.0f;
        }

        cudaMemcpyAsync(input_device, input_host.data(), input_size * sizeof(float), cudaMemcpyHostToDevice, stream);

        // void* bindings[] = {input_device, output_device};
        // context->enqueueV2(bindings, 0, nullptr);
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        cudaMemcpyAsync(output_host.data(), output_device, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

        for (size_t i = 0; i < output_size; ++i) {
            std::cout << "Output[" << i << "]: " << output_host[i] << std::endl;
        }

        cudaStreamDestroy(stream);
        cudaFree(input_device);
        cudaFree(output_device);

        return true;
    }

    bool run() {
        int ret = true;
        if (!std::filesystem::exists(_engine_path)) {
            ret = build();
        }
        ret = inference();
        return ret;
    }

// function alias
#define onnx2engine build
#define detect inference
#define infer inference

private:
    std::vector<uint8_t> load_file(std::filesystem::path path) {
        std::vector<uint8_t> data;
        std::ifstream file{path, std::ios::in | std::ios::binary};

        file.seekg(0, file.end);
        const auto size = file.tellg();
        file.seekg(0, file.beg);

        data.resize(size);
        file.read(reinterpret_cast<char*>(data.data()), size);
        file.close();
        return data;
    }

private:
    std::filesystem::path _onnx_path;
    std::filesystem::path _engine_path;
    nvinfer1::Dims _input_dims;
    nvinfer1::Dims _output_dims;
    std::shared_ptr<nvinfer1::ICudaEngine> _engine;
};

using Model = TensorRTModel;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Print helper
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::string tensor_shape(nvinfer1::ITensor* tensor) {
    std::string shape;
    auto dims = tensor->getDimensions();
    for (int i = 0; i < dims.nbDims; i++) {
        if (i == 0) {
            shape = fmt::format("{}", dims.d[i]);
        } else {
            shape = fmt::format("{}x{}", shape, dims.d[i]);
        }
    }
    return shape;
}

void print_network(nvinfer1::INetworkDefinition& network, nvinfer1::ICudaEngine& engine, bool optimized) {
    int input_count = network.getNbInputs();
    int output_count = network.getNbOutputs();
    std::string layer_info;

    for (int i = 0; i < input_count; i++) {
        auto input = network.getInput(i);
        AHRI_LOGGER_INFO("Input info: {}: {}", input->getName(), tensor_shape(input));
    }
    for (int i = 0; i < output_count; i++) {
        auto output = network.getInput(i);
        AHRI_LOGGER_INFO("Output info: {}: {}", output->getName(), tensor_shape(output));
    }

    int layer_count = optimized ? engine.getNbLayers() : network.getNbLayers();
    AHRI_LOGGER_INFO("network layer {}", layer_count);

    if (!optimized) {
        for (int i = 0; i < layer_count; i++) {
            auto layer = network.getLayer(i);
            auto input = layer->getInput(0);
            if (input == nullptr) {
                continue;
            }
            auto output = layer->getOutput(0);
            AHRI_LOGGER_INFO("Layer info: {}: {} {} {}", layer->getName(), tensor_shape(input), tensor_shape(output),
                             static_cast<int>(layer->getPrecision()));
        }
    } else {
        auto inspector = std::unique_ptr<nvinfer1::IEngineInspector>(engine.createEngineInspector());
        for (int i = 0; i < layer_count; i++) {
            AHRI_LOGGER_INFO("Layer info: {}",
                             inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }
}

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

/**
 * @brief Sample plugin impl
 */
namespace nvinfer1 {}  // namespace nvinfer1

#endif  // __NVCC__
#endif  // !TENSORRT_UTILS_CUH

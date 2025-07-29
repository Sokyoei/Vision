/**
 * @file tensorrt_utils.cuh
 * @date 2024/05/16
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef AHRI_VISION_TENSORRT_UTILS_HPP
#define AHRI_VISION_TENSORRT_UTILS_HPP

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Vision.hpp"

// #ifdef __NVCC__
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "Ahri.cuh"
#include "Ahri/Vision/tensorrt_macro.hpp"
#include "Ceceilia/utils/logger_utils.hpp"

#ifndef AHRI_CXX17
#error "requires compiler has C++17 or later."
#endif

// #define BUILD_SAMPLES

namespace Ahri::TensorRT {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void print_network(nvinfer1::INetworkDefinition& network, nvinfer1::ICudaEngine& engine, bool optimized);

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
inline void onnx_to_engine(std::string onnx_path, std::string engine_path, nvinfer1::BuilderFlag build_flag) {
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
inline void load_engine(const std::string& engine_path,
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
        parser->parseFromFile(_onnx_path.string().c_str(), 2);
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

class Buffer {
public:
    Buffer() {}
    ~Buffer() {}

private:
};

class TensorRTModel {
public:
    TensorRTModel(std::filesystem::path onnx_path) : _onnx_path(onnx_path) {
        if (!std::filesystem::exists(_onnx_path)) {
            AHRI_LOGGER_ERROR("{} not found.", _onnx_path);
        }
        _engine_path = _onnx_path;
        _engine_path.replace_extension(ENGINE_EXTENSION);

        cudaStreamCreate(&_stream);
    }

    ~TensorRTModel() { cudaStreamDestroy(_stream); }

    bool build() {
        // initLibNvInferPlugins(&trtlogger, "samples");

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtlogger));
        constexpr auto explicit_batch =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trtlogger));
        parser->parseFromFile(_onnx_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
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

        fmt::println("input_size: {}, output_size: {}", input_size, output_size);

        // alloc cpu and gpu memory
        // std::vector<float> input_host(input_size, 1.0);
        std::vector<float> input_host{
            0.7576, 0.2793, 0.4031, 0.7347, 0.0293,  //
            0.7999, 0.3971, 0.7544, 0.5695, 0.4388,  //
            0.6387, 0.5247, 0.6826, 0.3051, 0.4635,  //
            0.4550, 0.5725, 0.4980, 0.9371, 0.6556,  //
            0.3138, 0.1980, 0.4162, 0.2843, 0.3398,  //
        };
        std::vector<float> output_host(output_size, 0.0);
        void* input_device;
        void* output_device;
        cudaMalloc(&input_device, input_size * sizeof(float));
        cudaMalloc(&output_device, output_size * sizeof(float));

        context->setInputTensorAddress("input", input_device);
        context->setOutputTensorAddress("output", output_device);

        fmt::println("Input: {}", input_host);

        CUDA_CHECK(cudaMemcpyAsync(input_device, input_host.data(), input_size * sizeof(float), cudaMemcpyHostToDevice,
                                   _stream));

        // void* bindings[] = {input_device, output_device};
        // context->enqueueV2(bindings, 0, nullptr);
        context->enqueueV3(_stream);
        // cudaStreamSynchronize(stream);

        CUDA_CHECK(cudaMemcpyAsync(output_host.data(), output_device, output_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, _stream));
        cudaStreamSynchronize(_stream);

        fmt::println("Output: {}", output_host);

        CUDA_CHECK(cudaFree(input_device));
        CUDA_CHECK(cudaFree(output_device));
        input_device = nullptr;
        output_device = nullptr;

        return true;
    }

#ifdef USE_OPENCV
    std::vector<float> inference(const cv::Mat& image) {}
#endif  // USE_OPENCV

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

    std::map<std::string, nvinfer1::Weights> load_weights(std::filesystem::path path) {
        std::ifstream file{path};
        int32_t size;
        std::map<std::string, nvinfer1::Weights> weights;
        file >> size;
        if (size < 0) {
            AHRI_LOGGER_ERROR("No weights found in {}", path);
        }

        while (size > 0) {
            nvinfer1::Weights weight;
            std::string name;
            int weight_length;

            file >> name;
            file >> std::dec >> weight_length;

            uint32_t* values = new uint32_t[weight_length];
            for (int i = 0; i < weight_length; i++) {
                file >> std::hex >> values[i];
            }

            weight.type = nvinfer1::DataType::kFLOAT;
            weight.count = weight_length;
            weight.values = values;

            weights[name] = weight;
            size--;
        }

        return weights;
    }

private:
    std::filesystem::path _onnx_path;
    std::filesystem::path _engine_path;
    cudaStream_t _stream;
    nvinfer1::Dims _input_dims;
    nvinfer1::Dims _output_dims;
    std::shared_ptr<nvinfer1::ICudaEngine> _engine;
    nvinfer1::DataType _data_type;
};

using Model = TensorRTModel;

namespace examples {
inline bool build_weights(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> weights) {
    auto data = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 1, 5});
#if NV_TENSORRT_MAJOR >= 10
    auto linear_weight = network.addConstant(nvinfer1::Dims{}, weights["linear.weight"]);
    auto linear_weight_tensor = linear_weight->getOutput(0);
    auto fc = network.addMatrixMultiply(*data, nvinfer1::MatrixOperation::kNONE, *linear_weight_tensor,
                                        nvinfer1::MatrixOperation::kTRANSPOSE);
#else
    auto fc = network.addFullyConnected(*data, 1, weights["linear.weight"], {});
#endif
    fc->setName("linear1");
    fc->getOutput(0)->setName("output0");
    network.markOutput(*fc->getOutput(0));
    return true;
}
}  // namespace examples

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Print helper
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline std::string tensor_shape(nvinfer1::ITensor* tensor) {
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

inline void print_network(nvinfer1::INetworkDefinition& network, nvinfer1::ICudaEngine& engine, bool optimized) {
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
#define PLUGIN_GET_TYPE_OR_NAME_NAMESPACE_VERSION(name, namespace, version, Type_or_Name) \
    nvinfer1::AsciiChar const* getPlugin##Type_or_Name() const noexcept override {        \
        return name;                                                                      \
    }                                                                                     \
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {             \
        return namespace;                                                                 \
    }                                                                                     \
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {               \
        return version;                                                                   \
    }

/**
 * @details for @c nvinfer1::IPluginV2DynamicExt
 */
#define PLUGIN_GET_TYPE_NAMESPACE_VERSION(name, namespace, version) \
    PLUGIN_GET_TYPE_OR_NAME_NAMESPACE_VERSION(name, namespace, version, Type)

/**
 * @details for @c nvinfer1::IPluginCreator
 */
#define PLUGIN_GET_NAME_NAMESPACE_VERSION(name, namespace, version) \
    PLUGIN_GET_TYPE_OR_NAME_NAMESPACE_VERSION(name, namespace, version, Name)

/**
 * @details register plugin for dynamic library
 */
#define AHRI_REGISTER_TENSORRT_PLUGIN(creator)                                 \
    extern "C" AHRI_TENSORRT_API nvinfer1::IPluginCreator* create##creator() { \
        return new creator();                                                  \
    }
}  // namespace Ahri::TensorRT

#ifdef BUILD_SAMPLES
/**
 * @brief Sample plugin impl
 */

///
/// +-------------------------------+-----------------------------------+
/// | ~nvinfer1::IPlugin~           |                                   |
/// | nvinfer1::IPluginV2           | 单一 input/output                 |
/// | nvinfer1::IPluginV2Ext        | 单一 input and mix output         |
/// | nvinfer1::IPluginV2IOExt      | mix input/output implicit batch   |
/// | nvinfer1::IPluginV2DynamicExt | mix input/output dynamic shape    |
/// +-------------------------------+-----------------------------------+
///
// namespace samples {
#include <cuda_runtime.h>
#include <cmath>

__global__ void custom_scalar_kernel(const float* inputs,
                                     float* outputs,
                                     const float scalar,
                                     const float scale,
                                     const int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    outputs[index] = (inputs[index] + scalar) * scale;
}

void custom_scalar(const float* inputs,
                   float* outputs,
                   const float scalar,
                   const float scale,
                   const int elements,
                   cudaStream_t stream) {
    dim3 block_size(256, 1, 1);
    dim3 grid_size(std::ceil(float(elements) / 256), 1, 1);
    custom_scalar_kernel<<<grid_size, block_size, 0, stream>>>(inputs, outputs, scalar, scale, elements);
}

#define PLUGIN_NAME "Scalar"
#define PLUGIN_NAMESPACE ""
#define PLUGIN_VERSION "1"

class CustomScalarPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    CustomScalarPlugin() = delete;

    /**
     * @brief parse constructor
     */
    CustomScalarPlugin(const std::string& name, float scalar, float scale) : _name(name) {
        _params.scalar = scalar;
        _params.scale = scale;
    }

    /**
     * @brief clone and deserialize constructor
     */
    CustomScalarPlugin(const std::string& name, const void* buffer, size_t size) : _name(name) {
        memcpy(&_params, buffer, sizeof(_params));
    }

    ~CustomScalarPlugin() {}

    PLUGIN_GET_TYPE_NAMESPACE_VERSION(PLUGIN_NAME, PLUGIN_NAMESPACE, PLUGIN_VERSION)

    int32_t getNbOutputs() const noexcept override { return 1; }

    size_t getSerializationSize() const noexcept override { return sizeof(_params); }

    nvinfer1::DimsExprs getOutputDimensions(int32_t output_index,
                                            nvinfer1::DimsExprs const* inputs,
                                            int32_t nb_inputs,
                                            nvinfer1::IExprBuilder& expr_builder) noexcept override {
        return inputs[0];
    }

    nvinfer1::DataType getOutputDataType(int32_t index,
                                         nvinfer1::DataType const* input_types,
                                         int32_t nb_inputs) const noexcept override {
        return input_types[0];
    }

    void setPluginNamespace(char const* plugin_namespace) noexcept override { _namespace = plugin_namespace; }

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                            int32_t nb_inputs,
                            nvinfer1::PluginTensorDesc const* outputs,
                            int32_t nb_outputs) const noexcept override {
        return 0;
    }

    int32_t initialize() noexcept override { return 0; }

    void terminate() noexcept override {}

    void serialize(void* buffer) const noexcept override { memcpy(buffer, &_params, sizeof(_params)); }

    void destroy() noexcept override { delete this; }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc,
                    void const* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override {
        int elements = 1;
        for (int i = 0; i < input_desc[0].dims.nbDims; i++) {
            elements *= input_desc[0].dims.d[i];
        }

        custom_scalar(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), _params.scalar,
                      _params.scale, elements, stream);

        return 0;
    }

    IPluginV2DynamicExt* clone() const noexcept override {
        auto p = new CustomScalarPlugin(_name, &_params, sizeof(_params));
        p->setPluginNamespace(_namespace.c_str());
        return p;
    }

    bool supportsFormatCombination(int32_t pos,
                                   nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t nb_inputs,
                                   int32_t nb_outputs) noexcept override {
        switch (pos) {
            case 0:
                return in_out[0].type == nvinfer1::DataType::kFLOAT &&
                       in_out[0].format == nvinfer1::TensorFormat::kLINEAR;
            case 1:
                return in_out[1].type == nvinfer1::DataType::kFLOAT &&
                       in_out[1].format == nvinfer1::TensorFormat::kLINEAR;
            default:
                return false;
        }
        return false;
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                         int32_t nb_inputs,
                         nvinfer1::DynamicPluginTensorDesc const* out,
                         int32_t nb_outputs) noexcept override {}

    virtual void attachToContext(cudnnContext* /*cudnn*/,
                                 cublasContext* /*cublas*/,
                                 nvinfer1::IGpuAllocator* /*allocator*/) noexcept override {}

    void detachFromContext() noexcept override {}

private:
    const std::string _name;
    std::string _namespace;
    struct {
        float scalar;
        float scale;
    } _params;
};

class CustomScalarPluginCreator : public nvinfer1::IPluginCreator {
public:
    CustomScalarPluginCreator() {
        _plugin_attributes.emplace_back(
            nvinfer1::PluginField("scalar", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        _plugin_attributes.emplace_back(
            nvinfer1::PluginField("scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        _fc.nbFields = _plugin_attributes.size();
        _fc.fields = _plugin_attributes.data();
    }

    ~CustomScalarPluginCreator() {}

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override {
        float scalar = 0;
        float scale = 0;
        std::map<std::string, float*> params{
            {"scalar", &scalar},
            { "scale",  &scale},
        };

        for (int i = 0; i < fc->nbFields; i++) {
            if (params.find(fc->fields[i].name) != params.end()) {
                *params[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
            }
        }

        AHRI_LOGGER_INFO("Creating Scalar plugin with params: scalar={}, scale={}", scalar, scale);

        return new CustomScalarPlugin(name, scalar, scale);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* name,
                                           void const* serial_data,
                                           size_t serial_length) noexcept override {
        return new CustomScalarPlugin(name, serial_data, serial_length);
    }

    nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &_fc; }

    void setPluginNamespace(char const* plugin_namespace) noexcept override { _namespace = plugin_namespace; }

    PLUGIN_GET_NAME_NAMESPACE_VERSION(PLUGIN_NAME, PLUGIN_NAMESPACE, PLUGIN_VERSION)

private:
    std::string _namespace;
    // 保存这个插件所需要的权重和参数，从 ONNX 中获取，在 parse 时使用
    std::vector<nvinfer1::PluginField> _plugin_attributes;
    // 接受 nvinfer1::PluginField 传入进来的权重和参数，并将信息传递给 Plugin, 内部通过 createPlugin 来创建带参数的
    // Plugin
    nvinfer1::PluginFieldCollection _fc{};
};

REGISTER_TENSORRT_PLUGIN(CustomScalarPluginCreator);
// }  // namespace samples
#endif  // BUILD_SAMPLES

// #endif  // __NVCC__
#endif  // !AHRI_VISION_TENSORRT_UTILS_HPP

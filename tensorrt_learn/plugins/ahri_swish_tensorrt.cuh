/**
 * @brief Swish plugin using nvinfer1::IPluginV3/nvinfer1::IPluginV2DynamicExt
 */

#pragma once
#ifndef AHRI_SWISH_TENSORRT_HPP
#define AHRI_SWISH_TENSORRT_HPP

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <fmt/core.h>

#include "tensorrt_utils.cuh"

namespace Ahri::TensorRT::Plugin {
static constexpr char* swish_plugin_name{"AhriSwish"};
static constexpr char* swish_plugin_namespace{""};
static constexpr char* swish_plugin_version{"1"};

__global__ void swish_kernel(const float* inputs, float* outputs, const int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    outputs[index] = inputs[index] * (1.0f / (1.0f + std::exp(-inputs[index])));
}

void swish(const float* inputs, float* outputs, const int elements, cudaStream_t stream) {
    dim3 block_size(256, 1, 1);
    dim3 grid_size(std::ceil(float(elements / 256)), 1, 1);
    swish_kernel<<<grid_size, block_size, 0, stream>>>(inputs, outputs, elements);
}

namespace V3 {
class AhriSwishPlugin : public nvinfer1::IPluginV3,
                        public nvinfer1::IPluginV3OneBuildV2,
                        public nvinfer1::IPluginV3OneCore,
                        public nvinfer1::IPluginV3OneRuntime {
public:
    AhriSwishPlugin(int32_t nsample) : _nsample(nsample) {
        _plugin_attributes.clear();

        nvinfer1::PluginField pf_nsample{"nsample", &_nsample, nvinfer1::PluginFieldType::kINT32, 1};
        _plugin_attributes.emplace_back(pf_nsample);

        _pfc.nbFields = static_cast<int32_t>(_plugin_attributes.size());
        _pfc.fields = _plugin_attributes.data();
    }

    ~AhriSwishPlugin() {}

    // nvinfer1::IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        try {
            if (type == nvinfer1::PluginCapabilityType::kBUILD) {
                return static_cast<IPluginV3OneBuild*>(this);
            }
            if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
                return static_cast<IPluginV3OneRuntime*>(this);
            }
            assert(type == nvinfer1::PluginCapabilityType::kCORE);
            return static_cast<IPluginV3OneCore*>(this);
        } catch (...) {
            // log error
        }
        return nullptr;
    };

    nvinfer1::IPluginV3* clone() noexcept override { return new AhriSwishPlugin(_nsample); }

    // nvinfer1::IPluginV3OneBuildV2
    int32_t getAliasedInput(int32_t outputIndex) noexcept override { return -1; }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                            int32_t nbInputs,
                            nvinfer1::DynamicPluginTensorDesc const* out,
                            int32_t nbOutputs) noexcept override {
        return 0;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
                               int32_t nbOutputs,
                               const nvinfer1::DataType* inputTypes,
                               int32_t nbInputs) const noexcept override {  // 确保输入输出数量符合预期
        assert(nbInputs == _in_n);
        assert(nbOutputs == _out_n);

        assert(inputTypes[0] == nvinfer1::DataType::kFLOAT);

        // 设置输出类型为 int64
        outputTypes[0] = nvinfer1::DataType::kINT64;

        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs,
                            int32_t nbInputs,
                            nvinfer1::DimsExprs const* shapeInputs,
                            int32_t nbShapeInputs,
                            nvinfer1::DimsExprs* outputs,
                            int32_t nbOutputs,
                            nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        // 确保输入输出数量符合预期
        assert(nbInputs == _in_n);
        assert(nbOutputs == _out_n);

        // 输入张量的形状
        const nvinfer1::DimsExprs& inputShape = inputs[0];

        // 设置输出张量的形状，假设为 [BATCH, NUM_POINTS]
        outputs[0].nbDims = 2;                             // 输出维度数为 2
        outputs[0].d[0] = inputShape.d[0];                 // BATCH
        outputs[0].d[1] = exprBuilder.constant(_nsample);  // NUM_POINTS

        return 0;  // 返回 0 表示成功
    }

    bool supportsFormatCombination(int32_t pos,
                                   nvinfer1::DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs,
                                   int32_t nbOutputs) noexcept override {
        // 确保 pos 在合法范围内
        assert(pos < (nbInputs + nbOutputs));

        // 检查输入
        if (pos == 0) {
            // 输入必须是 float 类型，且格式为线性
            return inOut[pos].desc.type == nvinfer1::DataType::kFLOAT &&
                   inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
        }
        // 检查输出
        else if (pos == 1) {
            // 输出必须是 int64 类型，且格式为线性
            return inOut[pos].desc.type == nvinfer1::DataType::kINT64 &&
                   inOut[pos].desc.format == nvinfer1::TensorFormat::kLINEAR;
        }

        return false;  // 其他情况不支持
    }

    int32_t getNbOutputs() const noexcept override { return _out_n; }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs,
                            int32_t nbInputs,
                            nvinfer1::DynamicPluginTensorDesc const* outputs,
                            int32_t nbOutputs) const noexcept override {
        const int64_t B = inputs[0].desc.dims.d[0];
        const int64_t N = inputs[0].desc.dims.d[1];

        return B * N * sizeof(float);
    }

    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override { return 0; }

    int32_t getNbTactics() noexcept override { return 0; }

    char const* getTimingCacheID() noexcept override { return nullptr; }

    int32_t getFormatCombinationLimit() noexcept override { return kDEFAULT_FORMAT_COMBINATION_LIMIT; }

    char const* getMetadataString() noexcept override { return nullptr; }

    // nvinfer1::IPluginV3OneCore
    PLUGIN_GET_NAME_NAMESPACE_VERSION(swish_plugin_name, swish_plugin_namespace, swish_plugin_version)

    // nvinfer1::IPluginV3OneRuntime
    int32_t setTactic(int32_t tactic) noexcept override { return 0; }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in,
                          int32_t nbInputs,
                          nvinfer1::PluginTensorDesc const* out,
                          int32_t nbOutputs) noexcept override {
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                    nvinfer1::PluginTensorDesc const* outputDesc,
                    void const* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override {
        const float* xyz = static_cast<const float*>(inputs[0]);
        int64_t* idxs = static_cast<int64_t*>(outputs[0]);

        // 从PluginTensorDesc中提取维度信息
        int B = static_cast<int>(inputDesc[0].dims.d[0]);
        int N = static_cast<int>(inputDesc[0].dims.d[1]);
        int C = static_cast<int>(inputDesc[0].dims.d[2]);
        int S = static_cast<int>(outputDesc[0].dims.d[1]);

        // 计算所需的内存大小N
        size_t temp_size = B * N;
        size_t temp_size_byte = temp_size * sizeof(float);

        if (workspace == nullptr) {
            return -1;
        }
        cudaError_t err = cudaMemset(workspace, 10000, temp_size_byte);
        if (err != cudaSuccess) {
            return -1;
        }

        return 0;
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &_pfc; }

private:
    int32_t _nsample;
    nvinfer1::PluginFieldCollection _pfc;
    std::vector<nvinfer1::PluginField> _plugin_attributes;
    int32_t _in_n = 1;
    int32_t _out_n = 1;
};

class AhriSwishPluginCreater : public nvinfer1::IPluginCreatorV3One {
public:
    AhriSwishPluginCreater() {
        _plugin_attributes.clear();

        nvinfer1::PluginField pf_nsample = {"nsample", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        _plugin_attributes.emplace_back(pf_nsample);

        _pfc.nbFields = static_cast<int32_t>(_plugin_attributes.size());
        _pfc.fields = _plugin_attributes.data();
    }

    ~AhriSwishPluginCreater() {}

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const* name,
                                      nvinfer1::PluginFieldCollection const* fc,
                                      nvinfer1::TensorRTPhase phase) noexcept override {
        assert(fc->nbFields == 1);
        assert(fc->fields[0].type == nvinfer1::PluginFieldType::kINT32);
        fc->fields[0].name;

        AhriSwishPlugin* plugin = new AhriSwishPlugin(*static_cast<int32_t const*>(fc->fields[0].data));

        return plugin;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &_pfc; }

    PLUGIN_GET_NAME_NAMESPACE_VERSION(swish_plugin_name, swish_plugin_namespace, swish_plugin_version)

private:
    nvinfer1::PluginFieldCollection _pfc;
    std::vector<nvinfer1::PluginField> _plugin_attributes;
};
}  // namespace V3

inline namespace V2 {
class AHRI_TENSORRT_API AhriSwishPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    AhriSwishPlugin() = delete;

    /**
     * @brief parse constructor
     */
    AhriSwishPlugin(const std::string& name) : _name(name) {}

    /**
     * @brief clone and deserialize constructor
     */
    AhriSwishPlugin(const std::string& name, const void* buffer, size_t size) : _name(name) {}

    ~AhriSwishPlugin() {}

    PLUGIN_GET_TYPE_NAMESPACE_VERSION(swish_plugin_name, swish_plugin_namespace, swish_plugin_version)

    int32_t getNbOutputs() const noexcept override { return 1; }

    size_t getSerializationSize() const noexcept override { return 0; }

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

    void serialize(void* buffer) const noexcept override {}

    void destroy() noexcept override { delete this; }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc,
                    void const* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override {
        int elements = 1;
        for (int i = 0; i < input_desc[i].dims.nbDims; i++) {
            elements *= input_desc[i].dims.d[i];
        }

        swish(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), elements, stream);

        return 0;
    }

    IPluginV2DynamicExt* clone() const noexcept override {
        auto p = new AhriSwishPlugin(_name, nullptr, 0);
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
};

class AHRI_TENSORRT_API AhriSwishPluginCreater : public nvinfer1::IPluginCreator {
public:
    AhriSwishPluginCreater() {
        _fc.nbFields = _plugin_attributes.size();
        _fc.fields = _plugin_attributes.data();
    }

    ~AhriSwishPluginCreater() {}

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override {
        AHRI_LOGGER_INFO("Creating Swish plugin");
        return new AhriSwishPlugin(name);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* name,
                                           void const* serial_data,
                                           size_t serial_length) noexcept override {
        return new AhriSwishPlugin(name, serial_data, serial_length);
    }

    nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &_fc; }

    void setPluginNamespace(char const* plugin_namespace) noexcept override { _namespace = plugin_namespace; }

    PLUGIN_GET_NAME_NAMESPACE_VERSION(swish_plugin_name, swish_plugin_namespace, swish_plugin_version)

private:
    std::string _namespace;
    // 保存这个插件所需要的权重和参数，从 ONNX 中获取，在 parse 时使用
    std::vector<nvinfer1::PluginField> _plugin_attributes;
    // 接受 nvinfer1::PluginField 传入进来的权重和参数，并将信息传递给 Plugin, 内部通过 createPlugin 来创建带参数的
    // Plugin
    nvinfer1::PluginFieldCollection _fc{};
};
}  // namespace V2

AHRI_REGISTER_TENSORRT_PLUGIN(AhriSwishPluginCreater);
}  // namespace Ahri::TensorRT::Plugin

#endif  // !AHRI_SWISH_TENSORRT_HPP

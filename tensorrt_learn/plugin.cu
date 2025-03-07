#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>

#define PLUGIN_V3
#define PLUGIN_V2

namespace Ahri {
#if NV_TENSORRT_MAJOR > 8
#ifdef PLUGIN_V3
/**
 * @brief Plugin for V3
 */
#define ADD_PLUGIN_NAME "AddPlugin"
#define ADD_PLUGIN_NAMESPACE "Ahri"
#define ADD_PLUGIN_VERSION "0.0.1"

class AddPlugin : public nvinfer1::IPluginV3,
                  public nvinfer1::IPluginV3OneBuildV2,
                  public nvinfer1::IPluginV3OneCore,
                  public nvinfer1::IPluginV3OneRuntime {
public:
    AddPlugin(int32_t nsample) : _nsample(nsample) {
        _plugin_attributes.clear();

        nvinfer1::PluginField pf_nsample{"nsample", &_nsample, nvinfer1::PluginFieldType::kINT32, 1};
        _plugin_attributes.emplace_back(pf_nsample);

        _pfc.nbFields = static_cast<int32_t>(_plugin_attributes.size());
        _pfc.fields = _plugin_attributes.data();
    }

    ~AddPlugin() {}

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

    nvinfer1::IPluginV3* clone() noexcept override { return new AddPlugin(_nsample); }

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
    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return ADD_PLUGIN_NAME; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return ADD_PLUGIN_VERSION; }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ADD_PLUGIN_NAMESPACE; }

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

        // 调用内核函数
        // furthest_point_sampling_kernel_wrapper(B, N, S, xyz, static_cast<float*>(workspace), idxs, stream);

        // // 拷贝回主机方便调试
        // int64_t* idxs_d = new int64_t[ 100 ];
        // err             = cudaMemcpy( idxs_d, idxs, 100 * sizeof( int64_t ), cudaMemcpyDeviceToHost );
        // if ( err != cudaSuccess ) {
        //     return -1;
        // }
        // delete[] idxs_d;

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

class AddPluginCreater : public nvinfer1::IPluginCreatorV3One {
public:
    AddPluginCreater() {
        _plugin_attributes.clear();

        nvinfer1::PluginField pf_nsample = {"nsample", nullptr, nvinfer1::PluginFieldType::kINT32, 1};
        _plugin_attributes.emplace_back(pf_nsample);

        _pfc.nbFields = static_cast<int32_t>(_plugin_attributes.size());
        _pfc.fields = _plugin_attributes.data();
    }

    ~AddPluginCreater() {}

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const* name,
                                      nvinfer1::PluginFieldCollection const* fc,
                                      nvinfer1::TensorRTPhase phase) noexcept override {
        assert(fc->nbFields == 1);
        assert(fc->fields[0].type == nvinfer1::PluginFieldType::kINT32);
        fc->fields[0].name;

        AddPlugin* plugin = new AddPlugin(*static_cast<int32_t const*>(fc->fields[0].data));

        return plugin;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &_pfc; }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override { return ADD_PLUGIN_NAME; }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return ADD_PLUGIN_VERSION; }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return ADD_PLUGIN_NAMESPACE; }

private:
    nvinfer1::PluginFieldCollection _pfc;
    std::vector<nvinfer1::PluginField> _plugin_attributes;
};
#endif
// #else
#ifdef PLUGIN_V2
/**
 * @brief Plugin for V2
 */
class AddPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    // 构造函数
    AddPlugin(float value) : mValue(value) {}

    // 反序列化构造函数
    AddPlugin(const void* data, size_t length) {
        const char* d = static_cast<const char*>(data);
        const char* a = d;
        mValue = read<float>(d);
        assert(d == a + length);
    }

    // 克隆插件
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override { return new AddPlugin(mValue); }

    // 获取插件的名称
    const char* getPluginType() const noexcept override { return "AddPlugin"; }

    // 获取插件的版本
    const char* getPluginVersion() const noexcept override { return "1"; }

    // 获取插件的输出数量
    int getNbOutputs() const noexcept override { return 1; }

    // 配置插件
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                         int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out,
                         int nbOutputs) noexcept override {
        // 可以在这里进行插件的配置，如检查输入输出的维度等
    }

    // 计算输出的维度
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex,
                                            const nvinfer1::DimsExprs* inputs,
                                            int nbInputs,
                                            nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        nvinfer1::DimsExprs output(inputs[0]);
        return output;
    }

    // 序列化插件
    size_t getSerializationSize() const noexcept override { return sizeof(mValue); }

    // 序列化插件数据
    void serialize(void* buffer) const noexcept override {
        char* d = static_cast<char*>(buffer);
        const char* a = d;
        write(d, mValue);
        assert(d == a + getSerializationSize());
    }

    // 执行前向推理
    bool enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                 const nvinfer1::PluginTensorDesc* outputDesc,
                 const void* const* inputs,
                 void* const* outputs,
                 void* workspace,
                 cudaStream_t stream) noexcept override {
        const int count = inputDesc[0].dims.d[0];
        const float* input = static_cast<const float*>(inputs[0]);
        float* output = static_cast<float*>(outputs[0]);

        // 简单的加法操作
        for (int i = 0; i < count; ++i) {
            output[i] = input[i] + mValue;
        }

        return true;
    }

    // 获取插件的数据类型
    nvinfer1::DataType getOutputDataType(int index,
                                         const nvinfer1::DataType* inputTypes,
                                         int nbInputs) const noexcept override {
        return inputTypes[0];
    }

    // 销毁插件
    void destroy() noexcept override { delete this; }

    // 设置插件的命名空间
    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }

    // 获取插件的命名空间
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    // 其他未实现的方法，这里简单返回默认值
    bool supportsFormatCombination(int pos,
                                   const nvinfer1::PluginTensorDesc* inOut,
                                   int nbInputs,
                                   int nbOutputs) noexcept override {
        return true;
    }
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                            int nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs,
                            int nbOutputs) const noexcept override {
        return 0;
    }
    void attachToContext(cudnnContext* cudnnContext,
                         cublasContext* cublasContext,
                         nvinfer1::IGpuAllocator* gpuAllocator) noexcept override {}
    void detachFromContext() noexcept override {}

private:
    template <typename T>
    void write(char*& buffer, const T& val) const {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer) const {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    float mValue;
    std::string mNamespace;
};

class AddPluginCreator : public nvinfer1::IPluginCreator {
public:
    AddPluginCreator() {
        mPluginAttributes.emplace_back(nvinfer1::PluginField("value", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* getPluginName() const noexcept override { return "AddPlugin"; }

    const char* getPluginVersion() const noexcept override { return "1"; }

    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    nvinfer1::IPluginV2DynamicExt* createPlugin(const char* name,
                                                const nvinfer1::PluginFieldCollection* fc) noexcept override {
        const nvinfer1::PluginField* fields = fc->fields;
        float value = 0;
        for (int i = 0; i < fc->nbFields; ++i) {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "value")) {
                value = *static_cast<const float*>(fields[i].data);
            }
        }
        return new AddPlugin(value);
    }

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(const char* name,
                                                     const void* serialData,
                                                     size_t serialLength) noexcept override {
        return new AddPlugin(serialData, serialLength);
    }

    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC{};
};
#endif
#endif
}  // namespace Ahri

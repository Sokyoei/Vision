#include <string>

#include <onnxruntime_cxx_api.h>

namespace Ahri {
class AbstractONNXLoader {
public:
    AbstractONNXLoader(std::string& onnx_path) : _onnx_path(onnx_path) {
        auto env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5");
        const auto& api = Ort::GetApi();
        OrtTensorRTProviderOptionsV2* tensorrt_options;
        Ort::SessionOptions session_potions;
        session_potions.SetInterOpNumThreads(1);
        session_potions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    }
    virtual ~AbstractONNXLoader() {}
    virtual void interface() = 0;

private:
    std::string _onnx_path;
};
}  // namespace Ahri

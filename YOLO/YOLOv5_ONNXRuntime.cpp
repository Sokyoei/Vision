#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "config.h"
#include "yolo.hpp"

namespace Ahri {
class YOLOv5 {
public:
    YOLOv5(std::filesystem::path onnx_path) : _onnx_path(onnx_path) {
        _env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "YOLOv5"};
        _session_options.SetIntraOpNumThreads(1);
        _session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
#ifdef _WIN32
        _session = Ort::Session(_env, _onnx_path.wstring().c_str(), _session_options);
#else
        _session = Ort::Session(_env, _onnx_path.string().c_str(), _session_options);
#endif
        auto available_providers = Ort::GetAvailableProviders();
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_shape = _session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        for (auto&& i : input_shape) {
            std::cout << i << '\n';
        }
    }
    ~YOLOv5() {}
    void preprocess(cv::Mat& img) {
        cv::Mat dst;
        cv::cvtColor(img, dst, cv::COLOR_BGR2RGB);
    }
    std::vector<YOLOResult> inference(cv::Mat& img) {
        auto memory_info =
            Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        // auto output_tensors = _session->Run(Ort::RunOptions{nullptr}, , , 1, , 1);
    }
    void postprocess() {}
    void nms() {}

private:
    std::filesystem::path _onnx_path;
    Ort::Env _env;
    Ort::Session _session{nullptr};
    Ort::SessionOptions _session_options;
};
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    Ahri::YOLOv5 yolov5{std::filesystem::path(ROOT) / "models/yolov5s.onnx"};
    return 0;
}

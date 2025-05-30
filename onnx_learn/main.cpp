#include <filesystem>

#include "Vision.hpp"

#ifdef _WIN32
#include <Windows.h>
#else defined(__linux__)
#include <unistd.h>
#endif

#include "onnx_utils.hpp"

namespace Ahri::ONNX {
class YOLOV5ONNXRuntimeModel : public ONNXRuntimeModel {};
}  // namespace Ahri::ONNX

int main(int argc, char const* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    auto model = Ahri::ONNX::ONNXRuntimeModel(std::filesystem::path(VISION_ROOT) / "models/VGG.onnx");
    cv::Mat image{cv::Size(224, 224), CV_8UC3, cv::Scalar(0.5f)};
    cv::Mat image_float;
    // auto model = Ahri::ONNX::YOLOV5ONNXRuntimeModel(std::filesystem::path(VISION_ROOT) / "models/yolov5s.onnx");
    // cv::Mat input_image = cv::imread((std::filesystem::path(VISION_ROOT) / "images/bus.jpg").string());
    image.convertTo(image_float, CV_32F, 1.0 / 255.0);
    auto result = model.inference(image_float);

    return 0;
}

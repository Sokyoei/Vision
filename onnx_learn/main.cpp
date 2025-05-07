#include <filesystem>

#include "Vision.hpp"

#include "onnx_utils.hpp"

int main(int argc, char const* argv[]) {
    auto model = Ahri::ONNX::ONNXModel(std::filesystem::path(VISION_ROOT) / "onnx_learn/VGG.onnx");
    cv::Mat input_image{cv::Size(640, 640), CV_8UC3, cv::Scalar(0.5f)};
    model.inference(input_image);
    return 0;
}

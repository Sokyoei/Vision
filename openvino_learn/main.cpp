#include <filesystem>

#include "Vision.hpp"

#include <fmt/ranges.h>
#include <opencv2/opencv.hpp>

#include "Ahri/Vision/opencv_utils.hpp"
#include "Ahri/Vision/openvino_utils.hpp"
#include "Ahri/Vision/yolo.hpp"

namespace Ahri::OpenVINO::Samples {
void yolov5_example() {
    auto model_path = std::filesystem::path{VISION_ROOT} / "models/yolov5s.xml";
    auto model = Ahri::OpenVINO::OpenVINOModel{model_path};

    // cv::Mat img = cv::imread((std::filesystem::path{SOKYOEI_DATA_DIR} / "Ahri/Popstar Ahri.jpg").string());
    cv::Mat img = cv::imread((std::filesystem::path{VISION_ROOT} / "images/zidane.jpg").string());

    cv::Mat preprocess_img = Ahri::YOLO::preprocess(img);
    auto result = model.inference(preprocess_img);
    auto yolo_results = Ahri::YOLO::postprocess(result, img.cols, img.rows);

    Ahri::YOLO::plot(img, yolo_results);
    IMG_SHOW(img, "YOLO Detection Result", cv::WINDOW_FREERATIO)
}
}  // namespace Ahri::OpenVINO::Samples

int main(int argc, char const* argv[]) {
    fmt::println("OpenVINO Version: {}", ov::get_openvino_version());
    Ahri::OpenVINO::Samples::yolov5_example();

    return 0;
}

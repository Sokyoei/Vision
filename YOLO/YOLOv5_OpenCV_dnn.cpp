#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

namespace Ahri {
struct YOLOResult {
    int class_id;
    float score;
    cv::Rect box;
};

class YOLOv5Detector {
public:
    YOLOv5Detector(std::string onnx_path, int height, int width, float threshold_score)
        : _height(height), _width(width), _threshold_score(threshold_score) {
        _net = cv::dnn::readNetFromONNX(onnx_path);
    }
    ~YOLOv5Detector() {}

    void detect(cv::Mat& frame, std::vector<YOLOResult>& results) {
        int width = frame.cols;
        int height = frame.rows;
        int max_ = std::max(width, height);
        cv::Mat image = cv::Mat::zeros(cv::Size(max_, max_), CV_8UC3);
        cv::Rect roi(0, 0, width, height);
        frame.copyTo(image(roi));

        float x_factor = image.cols / 640.0f;
        float y_factor = image.rows / 640.0f;

        cv::Mat blob =
            cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(_width, _height), cv::Scalar(0, 0, 0), true, false);
        _net.setInput(blob);
        cv::Mat pred = _net.forward();
    }

private:
    cv::dnn::Net _net;
    float _threshold_score;
    int _height;
    int _width;
};
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    return 0;
}

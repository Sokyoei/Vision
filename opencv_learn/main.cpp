#include "opencv_utils.hpp"

int main(int argc, char const* argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
    CV_LOG_INFO(nullptr, fmt::format("OpenCV version: {}", cv::getVersionString()));

    Ahri::OpenCV::video_show(true, 0);
    return 0;
}

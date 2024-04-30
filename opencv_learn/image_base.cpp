#include <filesystem>

#include <opencv2/opencv.hpp>

#include "config.h"

int main(int argc, char const* argv[]) {
    std::string img_path = (std::filesystem::path(ROOT) / "data/Ahri/Popstar Ahri.jpg").u8string();

    cv::namedWindow("Popstar Ahri", cv::WINDOW_FREERATIO);
    cv::Mat src = cv::imread(img_path, cv::IMREAD_COLOR);
    cv::imshow("Popstar Ahri", src);
    cv::waitKey();
    cv::destroyAllWindows();
    return 0;
}

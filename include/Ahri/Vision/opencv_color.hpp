/**
 * @file color.hpp
 * @date 2024/07/15
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef AHRI_VISION_OPENCV_COLOR_HPP
#define AHRI_VISION_OPENCV_COLOR_HPP

#include <opencv2/opencv.hpp>

namespace Ahri::OpenCV {
#define RED cv::Scalar(0, 0, 255)
#define GREEN cv::Scalar(0, 255, 0)
#define BLUE cv::Scalar(255, 0, 0)
#define BLACK cv::Scalar(0, 0, 0)
#define WHITE cv::Scalar(255, 255, 255)
#define YELLOW cv::Scalar(0, 255, 255)
#define VIOLET cv::Scalar(238, 130, 238)
#define PINK cv::Scalar(203, 192, 255)
#define DEEPPINK cv::Scalar(147, 20, 255)
#define PURPLE cv::Scalar(128, 0, 128)
#define SKYBLUE cv::Scalar(230, 216, 173)
#define GOLD cv::Scalar(10, 215, 255)
#define DARKGRAY cv::Scalar(169, 169, 169)
}  // namespace Ahri::OpenCV

#endif  // !AHRI_VISION_OPENCV_COLOR_HPP

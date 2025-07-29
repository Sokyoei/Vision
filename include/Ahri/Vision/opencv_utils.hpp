/**
 * @file opencv_utils.hpp
 * @date 2024/07/17
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef AHRI_VISION_OPENCV_UTILS_HPP
#define AHRI_VISION_OPENCV_UTILS_HPP

#include <opencv2/opencv.hpp>

namespace Ahri::OpenCV {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define IMG_SHOW(img, winname, flag) \
    cv::namedWindow(winname, flag);  \
    cv::imshow(winname, img);        \
    cv::waitKey();                   \
    cv::destroyAllWindows();
}  // namespace Ahri::OpenCV

#endif  // !AHRI_VISION_OPENCV_UTILS_HPP

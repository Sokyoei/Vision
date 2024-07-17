/**
 * @file opencv_utils.hpp
 * @date 2024/07/17
 * @author Sokyoei
 *
 *
 */

#include <opencv2/opencv.hpp>

namespace Ahri {
#define IMG_SHOW(img, winname, flag) \
    cv::namedWindow(winname, flag);  \
    cv::imshow(winname, img);        \
    cv::waitKey();                   \
    cv::destroyWindow(winname);
}  // namespace Ahri

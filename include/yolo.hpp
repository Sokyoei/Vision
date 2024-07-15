/**
 * @file yolo.hpp
 * @date 2024/07/15
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef YOLO_HPP
#define YOLO_HPP

#include <vector>

#include <opencv2/opencv.hpp>

#include "color.hpp"

namespace Ahri {
struct YOLOResult {
    int class_id;
    float score;
    cv::Rect box;
};

enum class YOLOVersion { YOLOV3, YOLOV5, YOLOV7, YOLOX };

std::vector<YOLOResult> nms(std::vector<YOLOResult> results, float threshold);

void plot(cv::Mat& img, std::vector<YOLOResult> results) {
    for (auto&& result : results) {
        cv::rectangle(img, result.box, GREEN);
    }
}
}  // namespace Ahri

#endif  // !YOLO_HPP

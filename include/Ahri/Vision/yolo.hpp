/**
 * @file yolo.hpp
 * @date 2024/07/15
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef AHRI_VISION_YOLO_HPP
#define AHRI_VISION_YOLO_HPP

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Ahri/Vision/opencv_color.hpp"

namespace Ahri {
struct YOLOResult {
    int class_id;
    float score;
    cv::Rect box;
};

enum class YOLOVersion { YOLOV3, YOLOV5, YOLOV7, YOLOV8, YOLOV9, YOLOV10, YOLO_WORLD, YOLOX };

/**
 * @brief coco80 类别（YOLOV5 默认）
 */
std::vector<std::string> coco80{
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush",
};

std::vector<YOLOResult> nms(std::vector<YOLOResult> results, float threshold);

inline void plot(cv::Mat& img, std::vector<YOLOResult> results) {
    for (auto&& result : results) {
        cv::rectangle(img, result.box, GREEN);
    }
}
}  // namespace Ahri

#endif  // !AHRI_VISION_YOLO_HPP

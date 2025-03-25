/**
 * @file openvino_utils.hpp
 * @date 2024/10/09
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef OPENVINO_UTILS_HPP
#define OPENVINO_UTILS_HPP

namespace Ahri::OpenVINO {
class AbstractOpenVINOInference {
private:
    /* data */
public:
    AbstractOpenVINOInference(/* args */) {}
    ~AbstractOpenVINOInference() {}
};

class OpenVINOModel {
public:
    OpenVINOModel() {}
    ~OpenVINOModel() {}

private:
};

using Model = OpenVINOModel;
}  // namespace Ahri::OpenVINO

#endif  // !OPENVINO_UTILS_HPP

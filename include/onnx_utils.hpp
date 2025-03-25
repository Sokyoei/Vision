/**
 * @file onnx_utils.hpp
 * @date 2024/06/19
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef ONNX_UTILS_HPP
#define ONNX_UTILS_HPP

#include <filesystem>
#include <iostream>
#include <ranges>
#include <string>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace Ahri::ONNX {
class AbstractONNXRuntimeInference {
public:
    AbstractONNXRuntimeInference(std::filesystem::path onnx_path) : _onnx_path(onnx_path) {
        _env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "default"};
        _session_options.SetIntraOpNumThreads(1);
        _session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
#ifdef _WIN32
        _session = Ort::Session(_env, _onnx_path.wstring().c_str(), _session_options);
#else
        _session = Ort::Session(_env, _onnx_path.string().c_str(), _session_options);
#endif
        auto available_providers = Ort::GetAvailableProviders();

        // check for cuda
        auto index = std::ranges::find(available_providers, "CUDAExecutionProvider");

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_shape = _session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        for (auto&& i : input_shape) {
            std::cout << i << '\n';
        }
    }
    virtual ~AbstractONNXRuntimeInference() {}
    virtual void inference() = 0;

private:
    std::filesystem::path _onnx_path;
    Ort::Env _env;
    Ort::Session _session{nullptr};
    Ort::SessionOptions _session_options;
};

class ONNXModel {
public:
    ONNXModel(std::filesystem::path onnx_path) : _onnx_path(onnx_path) {
        _env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "default"};
        _session_options.SetIntraOpNumThreads(1);
        _session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
#ifdef _WIN32
        _session = Ort::Session(_env, _onnx_path.wstring().c_str(), _session_options);
#else
        _session = Ort::Session(_env, _onnx_path.string().c_str(), _session_options);
#endif
        auto available_providers = Ort::GetAvailableProviders();

        // check for cuda
        auto index = std::ranges::find(available_providers, "CUDAExecutionProvider");

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_shape = _session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        for (auto&& i : input_shape) {
            std::cout << i << '\n';
        }
    }

    ~ONNXModel() {}

private:
    std::filesystem::path _onnx_path;
    Ort::Env _env;
    Ort::Session _session{nullptr};
    Ort::SessionOptions _session_options;
};

using Model = ONNXModel;
}  // namespace Ahri::ONNX

#endif  // !ONNX_UTILS_HPP

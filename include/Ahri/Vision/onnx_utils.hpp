/**
 * @file onnx_utils.hpp
 * @date 2024/06/19
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef AHRI_VISION_ONNX_UTILS_HPP
#define AHRI_VISION_ONNX_UTILS_HPP

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

#include "Vision.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <onnxruntime_cxx_api.h>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "Ceceilia/utils/logger_utils.hpp"

namespace Ahri::ONNX {
class ONNXRuntimeModel {
public:
    ONNXRuntimeModel(std::filesystem::path onnx_path) : _onnx_path(onnx_path) {
        _env = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "default"};
        _session_options.SetIntraOpNumThreads(1);
        _session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

#ifdef _WIN32
        _session = Ort::Session(_env, _onnx_path.wstring().c_str(), _session_options);
#else
        _session = Ort::Session(_env, _onnx_path.string().c_str(), _session_options);
#endif

        auto available_providers = Ort::GetAvailableProviders();
        AHRI_LOGGER_INFO("Available providers: {}", available_providers);

        // check for cuda
        auto index = std::ranges::find(available_providers, "CUDAExecutionProvider");
        // Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(_session_options, 0));

        // get model input and output info(name and shape)
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = _session.GetInputCount();
        size_t num_output_nodes = _session.GetOutputCount();
        _input_names = std::vector<const char*>{num_input_nodes};
        _output_names = std::vector<const char*>{num_output_nodes};
        _input_shapes = std::vector<std::vector<int64_t>>{num_input_nodes};
        _output_shapes = std::vector<std::vector<int64_t>>{num_output_nodes};
        for (int i = 0; i < num_input_nodes; i++) {
            Ort::AllocatedStringPtr input_name = _session.GetInputNameAllocated(i, allocator);
            _input_names[i] = input_name.get();
            _input_shapes[i] = _session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            AHRI_LOGGER_INFO("Input {}: name={}, shape={}", i, _input_names[i], _input_shapes[i]);
        }
        for (int i = 0; i < num_output_nodes; i++) {
            Ort::AllocatedStringPtr output_name = _session.GetOutputNameAllocated(i, allocator);
            _output_names[i] = output_name.get();
            _output_shapes[i] = _session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            AHRI_LOGGER_INFO("Output {}: name={}, shape={}", i, _output_names[i], _output_shapes[i]);
        }
    }

    std::vector<Ort::Value> inference(std::vector<float>& input_data) {
        // int data_length = static_cast<int>(
        //     std::accumulate(_input_shapes[0].begin(), _input_shapes[0].end(), 1, std::multiplies<int64_t>()));
        // std::vector<float> input_data(data_length, 0.5f);
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_data.size(),
                                                                  _input_shapes[0].data(), _input_shapes[0].size());
        std::vector<Ort::Value> outputs =
            _session.Run(Ort::RunOptions{nullptr}, _input_names.data(), &input_tensor, 1, _output_names.data(), 1);
        return outputs;
    }

#ifdef USE_OPENCV
    std::vector<Ort::Value> inference(cv::Mat& image) {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor =
            Ort::Value::CreateTensor<uint8_t>(memory_info, image.data, image.total() * image.channels(),
                                              _input_shapes[0].data(), _input_shapes[0].size());
        std::vector<Ort::Value> outputs =
            _session.Run(Ort::RunOptions{nullptr}, _input_names.data(), &input_tensor, 1, _output_names.data(), 1);
        return outputs;
    }
#endif  // USE_OPENCV

    // virtual int preprocess(cv::Mat& input_image) = 0;

    // virtual int postprocess(std::vector<Ort::Value>& outputs) = 0;

    ~ONNXRuntimeModel() {
        _env.release();
        _session.release();
    }

private:
    std::filesystem::path _onnx_path;
    Ort::Env _env;
    Ort::Session _session{nullptr};
    Ort::SessionOptions _session_options;
    std::vector<const char*> _input_names;
    std::vector<const char*> _output_names;
    std::vector<std::vector<int64_t>> _input_shapes;
    std::vector<std::vector<int64_t>> _output_shapes;
};

using Model = ONNXRuntimeModel;

namespace Samples {
struct AhriNetResult {};

class AhriNet : public ONNXRuntimeModel {
    cv::Mat& preprocess(cv::Mat& input_image) {}

    std::vector<AhriNetResult> postprocess(std::vector<Ort::Value>& outputs) {}
};
}  // namespace Samples
}  // namespace Ahri::ONNX

#endif  // !AHRI_VISION_ONNX_UTILS_HPP

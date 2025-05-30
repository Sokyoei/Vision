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

#include <filesystem>
#include <memory>

#include "Vision.hpp"

#include <openvino/openvino.hpp>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "Ceceilia/utils/fmt.hpp"
#include "Ceceilia/utils/logger_utils.hpp"

namespace Ahri::OpenVINO {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define AHRI_OPENVINO_API AHRI_API

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Inference
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class OpenVINOModel {
public:
    /**
     * @param model_path ONNX or OpenVINO xml model file path
     */
    OpenVINOModel(std::filesystem::path model_path) : _model_path(model_path) {}

    ~OpenVINOModel() {}

    void build() {
        ov::Core core;
        std::shared_ptr<ov::Model> model = core.read_model(_model_path);
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    }

    void inference() {
        ov::Core core;

        // load plugin
        try {
#ifdef _WIN32
            core.add_extension("ahri_plugin_openvino.dll");
#elif defined(__linux__)
            core.add_extension("libahri_plugin_openvino.so");
#endif
        } catch (const std::runtime_error& e) {
            AHRI_LOGGER_ERROR("Load OpenVINO plugin fail, reason: {}", e.what());
        }

        std::shared_ptr<ov::Model> model = core.read_model(_model_path);
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        ov::Tensor input_tensor = infer_request.get_input_tensor();
        ov::Shape input_shape = input_tensor.get_shape();
        size_t input_size = ov::shape_size(input_shape);
        std::vector<float> input_data(input_size, 0.5);
        std::memcpy(input_tensor.data<float>(), input_data.data(), input_size * sizeof(float));

        infer_request.infer();

        ov::Tensor output_tensor = infer_request.get_output_tensor();
        ov::Shape output_shape = output_tensor.get_shape();
        size_t output_size = ov::shape_size(output_shape);
        std::vector<float> output_data(output_size);
        std::memcpy(output_data.data(), output_tensor.data<float>(), output_size * sizeof(float));

        for (auto&& i : output_data) {
            fmt::println("{}", i);
        }
    }

private:
    std::filesystem::path _model_path;
    ov::Core _core;
};

using Model = OpenVINOModel;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Print helper
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Samples {}  // namespace Samples
}  // namespace Ahri::OpenVINO

AHRI_FMT_FORMATTER_OSTREAM(ov::Version);

#endif  // !OPENVINO_UTILS_HPP

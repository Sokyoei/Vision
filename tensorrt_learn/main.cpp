#include <iostream>
#include <string>
#include <vector>

#include "Vision.hpp"

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fmt/ranges.h>
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "Ahri/Vision/opencv_utils.hpp"
#include "Ahri/Vision/tensorrt_utils.hpp"
#include "Ahri/Vision/yolo.hpp"

void yolov5_example() {
    auto model = Ahri::TensorRT::TensorRTModel(std::filesystem::path(VISION_ROOT) / "models/yolov5su.onnx");
    cv::Mat img = cv::imread((std::filesystem::path(VISION_ROOT) / "images/bus.jpg").string());
    auto preprocess_img = Ahri::YOLO::preprocess(img);
    model.build();
    std::vector<float> outputs = model.inference(preprocess_img);
    auto results = Ahri::YOLO::postprocess_yolov5u(outputs, img.cols, img.rows, 80, 0.25f, 0.7f);
    Ahri::YOLO::plot(img, results);
    IMG_SHOW(img, "ONNXRuntime YOLOV5SU Example", cv::WINDOW_FREERATIO);
}

int main(int argc, char const* argv[]) {
    fmt::println("CUDA version: {}.{}.{}", CUDA_VERSION / 1000, (CUDA_VERSION % 100) / 10, CUDA_VERSION % 10);
    fmt::println("TensorRT version: {}.{}.{}", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);

    initLibNvInferPlugins(&Ahri::TensorRT::trtlogger, "");

#ifdef _WIN32
    auto ahri_plugin_tensorrt = LoadLibraryEx(TEXT("ahri_plugin_tensorrt.dll"), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!ahri_plugin_tensorrt) {
        // 处理加载失败的情况
        DWORD error = GetLastError();
        std::cerr << "Failed to load plugin library. Error: " << error << std::endl;
        return 1;
    } else {
        std::cout << "Load DLL success" << std::endl;
    }
#elif defined(__linux__)
    void* ahri_plugin_tensorrt = dlopen("libahri_plugin_tensorrt.so", RTLD_LAZY);
#endif

    std::vector<float> input_host{
        0.7576, 0.2793, 0.4031, 0.7347, 0.0293,  //
        0.7999, 0.3971, 0.7544, 0.5695, 0.4388,  //
        0.6387, 0.5247, 0.6826, 0.3051, 0.4635,  //
        0.4550, 0.5725, 0.4980, 0.9371, 0.6556,  //
        0.3138, 0.1980, 0.4162, 0.2843, 0.3398,  //
    };
    fmt::println("Input: {}", input_host);

    auto model = Ahri::TensorRT::TensorRTModel(std::filesystem::path(VISION_ROOT) / "tensorrt_learn/ahrinet.onnx");
    model.build();
    std::vector<float> result = model.infer(input_host);

    fmt::println("Result: {}", result);

    // yolov5_example();

#ifdef _WIN32
    FreeLibrary(ahri_plugin_tensorrt);
#elif defined(__linux__)
    dlclose(ahri_plugin_tensorrt);
#endif

    return 0;
}

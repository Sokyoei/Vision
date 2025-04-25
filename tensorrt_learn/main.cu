#include <string>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>

// #include "tensorrt_learn/plugins/ahri_leakyrelu_tensorrt.cuh"
// #include "tensorrt_learn/plugins/ahri_swish_tensorrt.cuh"
#include "tensorrt_utils.cuh"

#include "Vision.h"

int main(int argc, char const* argv[]) {
    initLibNvInferPlugins(&Ahri::TensorRT::trtlogger, "");

#ifdef _WIN32
#elif defined(__linux__)
    void* ahri_plugin_tensorrt = dlopen("libahri_plugin_tensorrt.so", RTLD_LAZY);
#endif

    auto model = Ahri::TensorRT::TensorRTModel(std::filesystem::path(VISION_ROOT) / "tensorrt_learn/ahrinet.onnx");
    model.build();
    model.infer();

#ifdef _WIN32
#elif defined(__linux__)
    dlclose(ahri_plugin_tensorrt);
#endif

    return 0;
}

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
#include <cuda_runtime.h>

#include "Ahri/Vision/tensorrt_utils.hpp"

int main(int argc, char const* argv[]) {
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

    auto model = Ahri::TensorRT::TensorRTModel(std::filesystem::path(VISION_ROOT) / "tensorrt_learn/ahrinet.onnx");
    model.build();
    model.infer();

#ifdef _WIN32
    FreeLibrary(ahri_plugin_tensorrt);
#elif defined(__linux__)
    dlclose(ahri_plugin_tensorrt);
#endif

    return 0;
}

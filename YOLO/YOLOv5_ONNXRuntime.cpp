#include <iostream>

#include <onnxruntime_cxx_api.h>

int main(int argc, char const* argv[]) {
    auto env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5");
    const auto& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* tensorrt_options;
    Ort::SessionOptions session_potions;
    session_potions.SetInterOpNumThreads(1);
    session_potions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    return 0;
}

#include "tensorrt_utils.cuh"

int main(int argc, char const* argv[]) {
    Ahri::TensorRT::TensorRTModel model{R"(/media/supervisor/windowsd/Code/github/Vision/onnx_learn/VGG.onnx)"};
    model.build();
    model.infer();
    return 0;
}

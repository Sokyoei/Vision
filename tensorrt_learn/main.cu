#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>

#include "tensorrt_utils.cuh"

int main(int argc, char const* argv[]) {
    auto model = Ahri::TensorRT::Model("/media/supervisor/windowsd/Code/github/Vision/tensorrt_learn/ahrinet.onnx");
    model.build();
    model.infer();

    return 0;
}

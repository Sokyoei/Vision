#pragma once
#ifndef TENSORRT_LEARN_PLUGINS_KERNELS_LEAKYRELU_CUH
#define TENSORRT_LEARN_PLUGINS_KERNELS_LEAKYRELU_CUH

#include <cuda_runtime.h>

#include "Ahri/Vision/tensorrt_utils.hpp"

#ifdef __cplusplus
extern "C" {
#endif

AHRI_TENSORRT_API
void leakyrelu(const float* inputs, float* outputs, const float alpha, const int elements, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // !TENSORRT_LEARN_PLUGINS_KERNELS_LEAKYRELU_CUH

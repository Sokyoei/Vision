#pragma once
#ifndef AHRI_ASUKA_TENSOR_QUANT_HPP
#define AHRI_ASUKA_TENSOR_QUANT_HPP

#include <NvInfer.h>

/// TensorRT Quantization
/// +--------------------------------------+-------------+
/// | @b nvinfer1::IInt8EntropyCalibrator  |             |
/// | @b nvinfer1::IInt8EntropyCalibrator2 |             |
/// | @b nvinfer1::IInt8MinMaxCalibrator   |             |
/// +--------------------------------------+-------------+
/// TensorRT 10.1+
/// Explicit Quantization

namespace Ahri::TensorRT {
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
private:
public:
    Int8EntropyCalibrator() {}
    ~Int8EntropyCalibrator() {}
};
}  // namespace Ahri::TensorRT

#endif  // !AHRI_ASUKA_TENSOR_QUANT_HPP

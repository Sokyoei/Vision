#pragma once
#ifndef AHRI_VISION_TENSORRT_MACRO_HPP
#define AHRI_VISION_TENSORRT_MACRO_HPP

#include "Ahri/Ahri.cuh"

namespace Ahri::TensorRT {
#define ENGINE_EXTENSION ".engine"
#define TIMING_CACHE_EXTENSION ".timing_cache"
#define AHRI_TENSORRT_API AHRI_API
}  // namespace Ahri::TensorRT

#endif  // !AHRI_VISION_TENSORRT_MACRO_HPP

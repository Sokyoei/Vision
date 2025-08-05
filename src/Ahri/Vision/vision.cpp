#include "Vision.hpp"

#ifdef USE_OPENCV
#include "Ahri/Vision/opencv_color.hpp"
#include "Ahri/Vision/opencv_utils.hpp"
#endif

#ifdef USE_TENSORRT
#include "Ahri/Vision/tensorrt_macro.hpp"
#include "Ahri/Vision/tensorrt_utils.hpp"
#endif

#ifdef USE_ONNXRUNTIME
#include "Ahri/Vision/onnx_utils.hpp"
#endif

#ifdef USE_OPENVINO
#include "Ahri/Vision/openvino_utils.hpp"
#endif

#ifdef USE_REALSENSE2
#include "Ahri/Vision/realsense2_utils.hpp"
#endif

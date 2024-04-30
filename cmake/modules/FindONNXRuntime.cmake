# ONNXRuntime
#
# Variables:
#   ONNXRuntime_INCLUDE_DIR
#   ONNXRuntime_LIBRARY
#   onnxruntime
#   onnxruntime_providers_shared

unset(ONNXRuntime_INCLUDE_DIR CACHE)
find_path(
    ONNXRuntime_INCLUDE_DIR
    NAMES
    onnxruntime_c_api.h
    PATHS ${ONNXRuntime_DIR}/include/
)

unset(ONNXRuntime_LIBRARY CACHE)
if(WIN32)
    find_library(onnxruntime NAMES onnxruntime.lib PATHS ${ONNXRuntime_DIR}/lib/)
    find_library(onnxruntime_providers_shared NAMES onnxruntime_providers_shared.lib PATHS ${ONNXRuntime_DIR}/lib/)
    set(ONNXRuntime_LIBRARY onnxruntime;onnxruntime_providers_shared)
elseif(LINUX)
    find_library(onnxruntime NAMES onnxruntime.so PATHS ${ONNXRuntime_DIR}/lib/)
    find_library(onnxruntime_providers_shared NAMES onnxruntime_providers_shared.so PATHS ${ONNXRuntime_DIR}/lib/)
    set(ONNXRuntime_LIBRARY onnxruntime;onnxruntime_providers_shared)
endif(WIN32)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ONNXRuntime
    REQUIRED_VARS ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY onnxruntime onnxruntime_providers_shared
)

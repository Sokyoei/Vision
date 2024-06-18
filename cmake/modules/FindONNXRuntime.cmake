# ONNXRuntime
#
# Require:
#   -DONNXRuntime_ROOT_DIR=<your ONNXRuntime root dir>
#
# Variables:
#   ONNXRuntime_INCLUDE_DIRS
#   ONNXRuntime_LIBRARYS
#
# Libraries:
#   onnxruntime
#   onnxruntime_providers_shared
#   with cuda:
#       onnxruntime_providers_cuda
#       onnxruntime_providers_tensorrt

unset(ONNXRuntime_INCLUDE_DIRS CACHE)
find_path(
    ONNXRuntime_INCLUDE_DIRS
    NAMES onnxruntime_c_api.h onnxruntime_cxx_api.h
    PATHS ${ONNXRuntime_ROOT_DIR}/include/
)

unset(ONNXRuntime_LIBRARYS CACHE)
find_library(onnxruntime NAMES onnxruntime PATHS ${ONNXRuntime_ROOT_DIR}/lib/)
find_library(onnxruntime_providers_shared NAMES onnxruntime_providers_shared PATHS ${ONNXRuntime_ROOT_DIR}/lib/)
set(ONNXRuntime_LIBRARYS onnxruntime;onnxruntime_providers_shared)
if(CMAKE_CUDA_COMPILER)
    find_library(onnxruntime_providers_cuda NAMES onnxruntime_providers_cuda PATHS ${ONNXRuntime_ROOT_DIR}/lib/)
    find_library(onnxruntime_providers_tensorrt NAMES onnxruntime_providers_tensorrt PATHS ${ONNXRuntime_ROOT_DIR}/lib/)
    set(ONNXRuntime_LIBRARYS ${ONNXRuntime_LIBRARYS};onnxruntime_providers_cuda;onnxruntime_providers_tensorrt)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ONNXRuntime
    REQUIRED_VARS ONNXRuntime_ROOT_DIR ONNXRuntime_INCLUDE_DIRS ONNXRuntime_LIBRARYS
)
mark_as_advanced(ONNXRuntime_ROOT_DIR ONNXRuntime_INCLUDE_DIRS ONNXRuntime_LIBRARYS)

if(ONNXRuntime_FOUND)
    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        add_library(ONNXRuntime::ONNXRuntime INTERFACE IMPORTED)
        target_include_directories(ONNXRuntime::ONNXRuntime INTERFACE ${ONNXRuntime_INCLUDE_DIRS})
        target_link_directories(ONNXRuntime::ONNXRuntime INTERFACE ${ONNXRuntime_ROOT_DIR}/lib)
        target_link_libraries(ONNXRuntime::ONNXRuntime INTERFACE ${ONNXRuntime_LIBRARYS})
        # set_target_properties(
        #     ONNXRuntime::ONNXRuntime PROPERTIES
        #     INTERFACE_LINK_DIRECTORIES  ${ONNXRuntime_ROOT_DIR}/lib
        #     INTERFACE_LINK_LIBRARIES ${ONNXRuntime_LIBRARYS}
        #     INTERFACE_INCLUDE_DIRECTORIES ${ONNXRuntime_INCLUDE_DIRS}
        # )
    endif()
endif()

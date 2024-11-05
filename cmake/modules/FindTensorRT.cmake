# TensorRT
#
# Require:
#   -DTensorRT_ROOT_DIR=<your TensorRT root dir>
# Variables:
#   TensorRT_INCLUDE_DIR
#   TensorRT_LIBRARYS

if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "require cuda support")
endif()

# root
set(TensorRT_ROOT_DIR ${TensorRT_ROOT_DIR} CACHE PATH "TensorRT library root directory")

# include
unset(TensorRT_INCLUDE_DIR CACHE)
find_path(
    TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TensorRT_ROOT_DIR}/include/
    DOC "TensorRT include directory"
)

# version
unset(TensorRT_VERSION CACHE)
if(EXISTS ${TensorRT_INCLUDE_DIR}/NvInfer.h)
    file(STRINGS ${TensorRT_INCLUDE_DIR}/NvInferVersion.h NV_TENSORRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
    file(STRINGS ${TensorRT_INCLUDE_DIR}/NvInferVersion.h NV_TENSORRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
    file(STRINGS ${TensorRT_INCLUDE_DIR}/NvInferVersion.h NV_TENSORRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")
    string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_MAJOR ${NV_TENSORRT_MAJOR})
    string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_MINOR ${NV_TENSORRT_MINOR})
    string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_PATCH ${NV_TENSORRT_PATCH})
    set(TensorRT_VERSION ${TensorRT_MAJOR}.${TensorRT_MINOR}.${TensorRT_PATCH} CACHE STRING "TensorRT library version")
endif()

# library
unset(TensorRT_LIBRARYS CACHE)
find_library(nvinfer NAMES nvinfer PATHS ${TensorRT_ROOT_DIR}/lib/)
find_library(nvinfer_dispatch NAMES nvinfer_dispatch PATHS ${TensorRT_ROOT_DIR}/lib/)
find_library(nvinfer_lean NAMES nvinfer_lean PATHS ${TensorRT_ROOT_DIR}/lib/)
find_library(nvinfer_plugin NAMES nvinfer_plugin PATHS ${TensorRT_ROOT_DIR}/lib/)
find_library(nvinfer_vc_plugin NAMES nvinfer_vc_plugin PATHS ${TensorRT_ROOT_DIR}/lib/)
find_library(nvonnxparser NAMES nvonnxparser PATHS ${TensorRT_ROOT_DIR}/lib/)
find_library(nvparsers NAMES nvparsers PATHS ${TensorRT_ROOT_DIR}/lib/)
set(
    TensorRT_LIBRARYS
    nvinfer;nvinfer_dispatch;nvinfer_lean;nvinfer_plugin;nvinfer_vc_plugin;nvonnxparser;nvparsers
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    TensorRT
    REQUIRED_VARS TensorRT_ROOT_DIR TensorRT_INCLUDE_DIR TensorRT_LIBRARYS
)
mark_as_advanced(TensorRT_ROOT_DIR TensorRT_INCLUDE_DIR TensorRT_LIBRARYS)

if(TensorRT_FOUND)
    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT INTERFACE IMPORTED)
        target_include_directories(TensorRT::TensorRT INTERFACE ${TensorRT_INCLUDE_DIR})
        target_link_directories(TensorRT::TensorRT INTERFACE ${TensorRT_ROOT_DIR}/lib)
        target_link_libraries(TensorRT::TensorRT INTERFACE ${TensorRT_LIBRARYS})
    endif()
endif()

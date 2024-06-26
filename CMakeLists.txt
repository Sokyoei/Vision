cmake_minimum_required(VERSION 3.20.0)

project(Vision VERSION 0.0.1)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 20)

# project FindXXX.cmake PATH
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

option(ONNXRuntime "enable ONNXRuntime" ON)
option(TensorRT "enable TensorRT" OFF)

include(${PROJECT_SOURCE_DIR}/cmake/compiler.cmake)

set(ROOT ${PROJECT_SOURCE_DIR})

# config.h settings
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
configure_file(
    ${PROJECT_SOURCE_DIR}/config.h.cmake
    ${PROJECT_SOURCE_DIR}/config.h
)

if(ONNXRuntime)
    # WARNING: if default, the onnxruntime.dll first well load in C:\Windows\System32\onnxruntime.dll
    find_package(ONNXRuntime MODULE REQUIRED)
endif(ONNXRuntime)

find_package(OpenCV CONFIG REQUIRED)
add_subdirectory(opencv_learn)

if(TensorRT)
    find_package(TensorRT MODULE REQUIRED)
    add_subdirectory(tensorrt_learn)
endif(TensorRT)

add_subdirectory(YOLO)

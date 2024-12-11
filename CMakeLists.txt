cmake_minimum_required(VERSION 3.20.0)

project(Vision VERSION 0.0.1)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif(CMAKE_CUDA_COMPILER)

set(CMAKE_CXX_STANDARD 20)

# project FindXXX.cmake PATH
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

# options
option(ONNXRuntime "enable ONNXRuntime" ON)
option(TensorRT "enable TensorRT" OFF)
option(IntelRealSense "enable Intel IntelRealSense" OFF)
option(Application "enable Application" OFF)

include(${PROJECT_SOURCE_DIR}/cmake/compiler.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/library.cmake)

set(ROOT ${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

if(OpenCV_FOUND)
    add_subdirectory(opencv_learn)
endif(OpenCV_FOUND)

if(TensorRT AND TensorRT_FOUND)
    add_subdirectory(tensorrt_learn)
endif(TensorRT AND TensorRT_FOUND)

if(IntelRealSense)
    if(realsense2_FOUND)
        add_subdirectory(realsense_learn)
    endif(realsense2_FOUND)
endif(IntelRealSense)

add_subdirectory(YOLO)

# config.h settings
configure_file(
    ${PROJECT_SOURCE_DIR}/config.h.cmake
    ${PROJECT_SOURCE_DIR}/config.h
)

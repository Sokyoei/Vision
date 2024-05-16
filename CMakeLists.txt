cmake_minimum_required(VERSION 3.20.0)

project(Vision)

set(CMAKE_CXX_STANDARD 17)

# project FindXXX.cmake PATH
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)

option(ONNXRuntime "enable ONNXRuntime" ON)

include(${PROJECT_SOURCE_DIR}/cmake/compiler.cmake)

set(ROOT ${PROJECT_SOURCE_DIR})

# config.h settings
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)
configure_file(
    ${PROJECT_SOURCE_DIR}/config.h.cmake
    ${PROJECT_SOURCE_DIR}/config.h
)

if(ONNXRuntime)
    find_package(ONNXRuntime MODULE REQUIRED)
endif(ONNXRuntime)

find_package(OpenCV CONFIG REQUIRED)
add_subdirectory(opencv_learn)

add_subdirectory(YOLO)

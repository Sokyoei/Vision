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
option(IntelRealSense "enable Intel IntelRealSense" OFF)
option(OpenGL_Glad "enable OpenGL Glad" OFF)

include(${PROJECT_SOURCE_DIR}/cmake/compiler.cmake)

set(ROOT ${PROJECT_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

# thrid libraries
find_package(fmt CONFIG REQUIRED)
# target_link_libraries(main PRIVATE fmt::fmt)
# target_link_libraries(main PRIVATE fmt::fmt-header-only)
if(fmt_FOUND)
    set(USE_FMT true)
endif(fmt_FOUND)

if(ONNXRuntime)
    # WARNING: if default, the onnxruntime.dll first well load in C:\Windows\System32\onnxruntime.dll
    find_package(ONNXRuntime MODULE REQUIRED)
    if(ONNXRuntime_FOUND)
        set(USE_ONNXRUNTIME true)
    endif()
endif(ONNXRuntime)

find_package(OpenCV CONFIG REQUIRED)
if(OpenCV_FOUND)
    set(USE_OPENCV true)
    add_subdirectory(opencv_learn)
endif()

find_package(FFMPEG REQUIRED)
# target_include_directories(main PRIVATE ${FFMPEG_INCLUDE_DIRS})
# target_link_directories(main PRIVATE ${FFMPEG_LIBRARY_DIRS})
# target_link_libraries(main PRIVATE ${FFMPEG_LIBRARIES})
if(FFMPEG_FOUND)
    set(USE_FFMPEG true)
    add_subdirectory(ffmpeg_learn)
endif(FFMPEG_FOUND)

if(TensorRT)
    find_package(TensorRT MODULE REQUIRED)
    if(TensorRT_FOUND)
        set(USE_TENSORRT true)
        add_subdirectory(tensorrt_learn)
    endif()
endif(TensorRT)

if(IntelRealSense)
    find_package(realsense2 CONFIG REQUIRED)
    # target_link_libraries(main PRIVATE realsense2::fw realsense2::rsutils realsense2::realsense2 realsense2::realsense-file)
    if(realsense2_FOUND)
        set(USE_REALSENSE2 true)
        add_subdirectory(realsense_learn)
    endif(realsense2_FOUND)
endif(IntelRealSense)

if(OpenGL_Glad)
    find_package(glad CONFIG REQUIRED)
    # target_link_libraries(main PRIVATE glad::glad)
    find_package(glfw3 CONFIG REQUIRED)
    # target_link_libraries(main PRIVATE glfw)
    if(glad_FOUND)
        set(USE_GLAD true)
    endif(glad_FOUND)
    if(glfw3_FOUND)
        set(USE_GLFW3 true)
    endif(glfw3_FOUND)
    if((glad_FOUND) AND (glfw3_FOUND))
        add_subdirectory(opengl_learn)
    endif((glad_FOUND) AND (glfw3_FOUND))
endif(OpenGL_Glad)

add_subdirectory(YOLO)

# config.h settings
configure_file(
    ${PROJECT_SOURCE_DIR}/config.h.cmake
    ${PROJECT_SOURCE_DIR}/config.h
)

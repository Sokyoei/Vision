add_executable(base base.cu)
target_link_libraries(base PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)

add_executable(builder builder.cu)
target_link_libraries(builder PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)
target_link_libraries(builder PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)

file(GLOB_RECURSE KERNEL_FILES plugins/kernels/*.cu)
add_library(ahri_plugin_tensorrt SHARED plugins/ahri_plugin_tensorrt.cpp ${KERNEL_FILES})
target_link_libraries(ahri_plugin_tensorrt PRIVATE TensorRT::TensorRT)
target_include_directories(ahri_plugin_tensorrt PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(ahri_plugin_tensorrt PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(ahri_plugin_tensorrt PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)
target_link_libraries(ahri_plugin_tensorrt PRIVATE opencv_core opencv_imgproc opencv_highgui)
# target_compile_options(ahri_plugin_tensorrt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC -Xcompiler=-fvisibility=default>)

add_executable(plugin plugin.cu)
target_link_libraries(plugin PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)
target_link_libraries(plugin PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)

add_executable(inference inference.cu)
target_link_libraries(inference PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)

add_executable(tensorrt_main main.cpp)
target_link_libraries(tensorrt_main PRIVATE TensorRT::TensorRT)
target_include_directories(tensorrt_main PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(tensorrt_main PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(tensorrt_main PRIVATE ahri_plugin_tensorrt)
target_link_libraries(tensorrt_main PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)
target_link_libraries(tensorrt_main PRIVATE opencv_core opencv_imgproc opencv_highgui)
set_target_properties(tensorrt_main PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN"
)

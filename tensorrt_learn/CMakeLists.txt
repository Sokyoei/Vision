add_executable(base base.cu)
target_link_libraries(base PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)

add_executable(builder builder.cu)
target_link_libraries(builder PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)
target_link_libraries(builder PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)

add_library(ahri_plugin_tensorrt SHARED plugins/ahri_plugin_tensorrt.cu)
target_link_libraries(ahri_plugin_tensorrt PRIVATE TensorRT::TensorRT)
target_link_libraries(ahri_plugin_tensorrt PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)

add_executable(plugin plugin.cu)
target_link_libraries(plugin PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)
target_link_libraries(plugin PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)

add_executable(inference inference.cu)
target_link_libraries(inference PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)

add_executable(base base.cu)
target_link_libraries(base PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)

add_executable(builder builder.cu)
target_link_libraries(builder PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)
target_link_libraries(builder PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)

add_executable(inference inference.cu)
target_link_libraries(inference PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)

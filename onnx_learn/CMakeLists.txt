add_executable(onnx_main main.cpp)
target_link_libraries(onnx_main PRIVATE ONNXRuntime::ONNXRuntime)
target_link_libraries(onnx_main PRIVATE opencv_core opencv_imgproc opencv_highgui)
target_link_libraries(onnx_main PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)

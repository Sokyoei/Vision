add_executable(base base.cu)
target_link_libraries(base PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)

add_executable(inference inference.cu)
target_link_libraries(inference PRIVATE TensorRT::TensorRT opencv_core opencv_imgproc)

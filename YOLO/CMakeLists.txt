add_executable(YOLOv5_ONNXRuntime YOLOv5_ONNXRuntime.cpp)
target_link_libraries(YOLOv5_ONNXRuntime PRIVATE ONNXRuntime::ONNXRuntime)
target_link_libraries(YOLOv5_ONNXRuntime PRIVATE opencv_core opencv_imgproc opencv_highgui)

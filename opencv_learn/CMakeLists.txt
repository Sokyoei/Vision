add_executable(image_base image_base.cpp)
target_link_libraries(image_base PRIVATE opencv_core opencv_imgproc opencv_highgui)

add_executable(camera camera.cpp)
target_link_libraries(camera PRIVATE opencv_core opencv_imgproc opencv_highgui)

add_executable(realsense_base base.cpp)
target_link_libraries(realsense_base PRIVATE realsense2::fw realsense2::rsutils realsense2::realsense2 realsense2::realsense-file)

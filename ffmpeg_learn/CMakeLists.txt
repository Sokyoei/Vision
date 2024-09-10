add_executable(ffmpeg_learn ffmpeg_learn.cpp)
target_include_directories(ffmpeg_learn PRIVATE ${FFMPEG_INCLUDE_DIRS})
target_link_directories(ffmpeg_learn PRIVATE ${FFMPEG_LIBRARY_DIRS})
target_link_libraries(ffmpeg_learn PRIVATE ${FFMPEG_LIBRARIES})

add_executable(opengl_learn opengl_learn.cpp)
target_link_libraries(opengl_learn PRIVATE glad::glad)
target_link_libraries(opengl_learn PRIVATE glfw)

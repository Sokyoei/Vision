add_library(ahri_plugin_openvino SHARED plugins/ahri_plugin_openvino.cpp)
target_link_libraries(ahri_plugin_openvino PRIVATE openvino::runtime)

add_executable(openvino_main main.cpp)
target_link_libraries(openvino_main PRIVATE openvino::runtime)
target_link_libraries(openvino_main PRIVATE fmt::fmt-header-only spdlog::spdlog_header_only)

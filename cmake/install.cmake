########################################################################################################################
# install
########################################################################################################################
install(DIRECTORY include/ DESTINATION include)

# if(TARGET ${PROJECT_NAME})
#     install(
#         EXPORT ${PROJECT_NAME}Targets
#         FILE ${PROJECT_NAME}Targets.cmake
#         # NAMESPACE ${PROJECT_NAME}::
#         NAMESPACE Ahri::
#         DESTINATION ${CMAKE_INSTALL_LIB_DIR}/cmake/${PROJECT_NAME}
#     )
# endif()

message(STATUS "all install executables: ${ALL_INSTALL_EXECUTABLES}")

install(
    TARGETS ${ALL_INSTALL_EXECUTABLES}
    # RUNTIME_DEPENDENCY_SET runtime_deps
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    # CONFIGURATIONS Release
)

# install(
#     RUNTIME_DEPENDENCY_SET runtime_deps
#     DESTINATION bin
#     PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-"
#     POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
# )

# the vcpkg can copy dlls to compile directory, then we copy them to bin directory
if(WIN32)
    file(
        GLOB DEPENDS_DLLS
        "${CMAKE_CURRENT_BINARY_DIR}/*.dll"
        "${CMAKE_CURRENT_BINARY_DIR}/*/*.dll"
    )
    set(INSTALL_DEPENDS_DIR bin)
else()
    file(
        GLOB DEPENDS_DLLS
        "${CMAKE_CURRENT_BINARY_DIR}/*.so"
        "${CMAKE_CURRENT_BINARY_DIR}/*.so.*"
        "${CMAKE_CURRENT_BINARY_DIR}/*/*.so"
        "${CMAKE_CURRENT_BINARY_DIR}/*/*.so.*"
    )
    set(INSTALL_DEPENDS_DIR lib)
endif(WIN32)

if(DEPENDS_DLLS)
    list(REMOVE_DUPLICATES DEPENDS_DLLS)
    install(FILES ${DEPENDS_DLLS} DESTINATION ${INSTALL_DEPENDS_DIR})
    foreach(dll ${DEPENDS_DLLS})
        get_filename_component(dll_name ${dll} NAME)
        message(STATUS "Installed DLL: ${dll_name} to ${INSTALL_DEPENDS_DIR}")
    endforeach()
endif()

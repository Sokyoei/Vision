set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")

macro(install_exe exe)
    install(
        TARGETS ${exe}
        RUNTIME_DEPENDENCY_SET runtime_deps
        RUNTIME
        DESTINATION bin
    )
endmacro(install_exe)

macro(install_lib lib)
    install(
        TARGETS ${lib}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
endmacro(install_lib)

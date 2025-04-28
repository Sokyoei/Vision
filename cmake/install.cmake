set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")

macro(install_${PROJECT_NAME}_exe exe)
    install(TARGETS ${exe} RUNTIME DESTINATION bin)
endmacro(install_${PROJECT_NAME}_exe)

macro(install_${PROJECT_NAME}_lib lib)
    install(
        TARGETS ${lib}
        EXPORT ${PROJECT_NAME}Targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
    )
endmacro(install_${PROJECT_NAME}_lib)

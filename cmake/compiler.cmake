# Global
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MSVC
set(MSVC_BASE_FLAGS "/EHsc /utf-8")
set(MSVC_CC_FLAGS "${MSVC_BASE_FLAGS} /Zc:__STDC__")
set(MSVC_CXX_FLAGS "${MSVC_BASE_FLAGS} /Zc:__cplusplus")

# GCC
set(GCC_BASE_FLAGS "-fdiagnostics-color=always -Wall")
set(GCC_CC_FLAGS "${GCC_BASE_FLAGS}")
set(GCC_CXX_FLAGS "${GCC_BASE_FLAGS}")

# Windows China
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
    execute_process(
        COMMAND chcp.com
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE chcp_out
    )
    if(${chcp_out} MATCHES "936")
        set(GCC_CC_FLAGS "${GCC_CC_FLAGS} -fexec-charset=gbk")
        set(GCC_CXX_FLAGS "${GCC_CXX_FLAGS} -fexec-charset=gbk")
    endif()
endif()

if(CMAKE_C_COMPILER_LOADED)
    if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_C_FLAGS ${MSVC_CC_FLAGS})
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_C_FLAGS ${GCC_CC_FLAGS})
    endif()
    message(STATUS "CC: ${CMAKE_C_COMPILER_ID}, FLAGS: ${CMAKE_C_FLAGS}")
endif()
if(CMAKE_CXX_COMPILER_LOADED)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_FLAGS ${MSVC_CXX_FLAGS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS ${GCC_CXX_FLAGS})
    endif()
    message(STATUS "CXX: ${CMAKE_CXX_COMPILER_ID}, FLAGS: ${CMAKE_CXX_FLAGS}")
endif()

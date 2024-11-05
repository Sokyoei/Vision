# ONNXRuntime
#
# Require:
#   -DONNXRuntime_ROOT_DIR=<your ONNXRuntime root dir>
#
# Variables:
#   ONNXRuntime_ROOT_DIR
#   ONNXRuntime_VERSION
#   ONNXRuntime_INCLUDE_DIR
#   ONNXRuntime_LIBRARYS
#
# Libraries:
#   onnxruntime
#   onnxruntime_providers_shared
#   with cuda:
#       onnxruntime_providers_cuda
#       onnxruntime_providers_tensorrt

# root
set(ONNXRuntime_ROOT_DIR ${ONNXRuntime_ROOT_DIR} CACHE PATH "ONNXRuntime library root directory")

# include
unset(ONNXRuntime_INCLUDE_DIR CACHE)
find_path(
    ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_c_api.h onnxruntime_cxx_api.h
    PATHS ${ONNXRuntime_ROOT_DIR}/include/
    DOC "ONNXRuntime include directory"
)

# version
unset(ONNXRuntime_VERSION CACHE)
if(EXISTS ${ONNXRuntime_INCLUDE_DIR}/onnxruntime_c_api.h)
    file(STRINGS ${ONNXRuntime_INCLUDE_DIR}/onnxruntime_c_api.h ORT_API_VERSION REGEX "^#define ORT_API_VERSION [0-9]+.*$")
    string(REGEX REPLACE "^#define ORT_API_VERSION ([0-9]+).*$" "\\1" ONNXRuntime_VERSION ${ORT_API_VERSION})
    set(ONNXRuntime_VERSION ${ONNXRuntime_VERSION} CACHE STRING "ONNXRuntime library version")
endif()

# library
unset(ONNXRuntime_LIBRARYS CACHE)
find_library(onnxruntime NAMES onnxruntime PATHS ${ONNXRuntime_ROOT_DIR}/lib/)
find_library(onnxruntime_providers_shared NAMES onnxruntime_providers_shared PATHS ${ONNXRuntime_ROOT_DIR}/lib/)
set(ONNXRuntime_LIBRARYS onnxruntime;onnxruntime_providers_shared)
if(CMAKE_CUDA_COMPILER)
    find_library(onnxruntime_providers_cuda NAMES onnxruntime_providers_cuda PATHS ${ONNXRuntime_ROOT_DIR}/lib/)
    find_library(onnxruntime_providers_tensorrt NAMES onnxruntime_providers_tensorrt PATHS ${ONNXRuntime_ROOT_DIR}/lib/)
    set(ONNXRuntime_LIBRARYS ${ONNXRuntime_LIBRARYS};onnxruntime_providers_cuda;onnxruntime_providers_tensorrt)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ONNXRuntime
    REQUIRED_VARS ONNXRuntime_ROOT_DIR ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARYS
)
mark_as_advanced(ONNXRuntime_ROOT_DIR ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARYS)

if(ONNXRuntime_FOUND)
    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        add_library(ONNXRuntime::ONNXRuntime INTERFACE IMPORTED)
        target_include_directories(ONNXRuntime::ONNXRuntime INTERFACE ${ONNXRuntime_INCLUDE_DIR})
        target_link_directories(ONNXRuntime::ONNXRuntime INTERFACE ${ONNXRuntime_ROOT_DIR}/lib)
        target_link_libraries(ONNXRuntime::ONNXRuntime INTERFACE ${ONNXRuntime_LIBRARYS})
        # set_target_properties(
        #     ONNXRuntime::ONNXRuntime PROPERTIES
        #     INTERFACE_LINK_DIRECTORIES  ${ONNXRuntime_ROOT_DIR}/lib
        #     INTERFACE_LINK_LIBRARIES ${ONNXRuntime_LIBRARYS}
        #     INTERFACE_INCLUDE_DIRECTORIES ${ONNXRuntime_INCLUDE_DIR}
        # )
    endif()
endif()

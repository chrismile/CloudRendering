cmake_minimum_required(VERSION 3.11)
project(.)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.c src/*.h src/*.hpp)
file(GLOB_RECURSE CUDA_SOURCES src/*.cu src/*.cuh)

include_directories(src)

if(MSVC)
    set(CMAKE_CXX_FLAGS "-W3 /EHsc")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-Wall")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-Wall")
endif()

if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    # https://cmake.org/cmake/help/git-stage/module/FindCUDA.html
    find_package(CUDA QUIET)
    if(${CUDA_FOUND})
        add_library(. SHARED ${SOURCES} ${CUDA_SOURCES})
    else()
        add_library(. SHARED ${SOURCES})
    endif()
else()
    # https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html
    find_package(CUDAToolkit QUIET)
    if(${CUDAToolkit_FOUND})
        add_library(. SHARED ${SOURCES} ${CUDA_SOURCES})
    else()
        add_library(. SHARED ${SOURCES})
    endif()
endif()

cmake_policy(SET CMP0012 NEW)
find_package(Torch REQUIRED)

target_link_libraries(. ${TORCH_LIBRARIES})
target_include_directories(. PUBLIC ${TORCH_INCLUDE_DIR})

if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
    # https://cmake.org/cmake/help/git-stage/module/FindCUDA.html
    if(${CUDA_FOUND})
        enable_language(CUDA)
        set_target_properties(. PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_compile_definitions(USE_CUDA)
    endif()
else()
    # https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html
    if(${CUDAToolkit_FOUND})
        enable_language(CUDA)
        # https://en.wikipedia.org/wiki/CUDA
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
        #set_property(TARGET kpn_module PROPERTY CUDA_ARCHITECTURES 61 75 86)
        set_property(TARGET . PROPERTY CUDA_ARCHITECTURES 61 75)
        set_target_properties(. PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_compile_definitions(USE_CUDA)
    endif()
endif()


install(
        TARGETS .
        ARCHIVE DESTINATION modules
        LIBRARY DESTINATION modules
        RUNTIME DESTINATION bin
)

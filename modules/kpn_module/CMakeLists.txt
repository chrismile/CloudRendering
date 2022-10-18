cmake_minimum_required (VERSION 3.7)
cmake_policy(VERSION 3.7...3.20)
project(kpn_module)
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
else()
    # https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html
    find_package(CUDAToolkit QUIET)
endif()

if(CUDAToolkit_FOUND OR CUDA_FOUND)
    add_library(kpn_module SHARED ${SOURCES} ${CUDA_SOURCES})
    enable_language(CUDA)
    set_target_properties(kpn_module PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(kpn_module PRIVATE USE_CUDA)
else()
    add_library(kpn_module SHARED ${SOURCES})
endif()

if(CUDAToolkit_FOUND)
    # https://en.wikipedia.org/wiki/CUDA
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability
    if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.1")
        set_property(TARGET kpn_module PROPERTY CUDA_ARCHITECTURES 61 75 86)
    elseif (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "10.0")
        set_property(TARGET kpn_module PROPERTY CUDA_ARCHITECTURES 61 75)
    else()
        set_property(TARGET kpn_module PROPERTY CUDA_ARCHITECTURES 61)
    endif()
    set_target_properties(vpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()


if (WIN32)
    set(DLLIMPORT "__declspec(dllimport)")
    set(DLLEXPORT "__declspec(dllexport)")
    target_compile_definitions(kpn_module INTERFACE MODULE_OP_API=${DLLIMPORT})
    target_compile_definitions(kpn_module PRIVATE MODULE_OP_API=${DLLEXPORT})
else()
    target_compile_definitions(kpn_module PUBLIC MODULE_OP_API=)
endif()


cmake_policy(SET CMP0012 NEW)
find_package(Torch REQUIRED)

target_link_libraries(kpn_module ${TORCH_LIBRARIES})
target_include_directories(kpn_module PUBLIC ${TORCH_INCLUDE_DIR})


install(
        TARGETS kpn_module
        ARCHIVE DESTINATION modules
        LIBRARY DESTINATION modules
        RUNTIME DESTINATION bin
)

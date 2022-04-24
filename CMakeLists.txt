cmake_minimum_required (VERSION 3.7)
cmake_policy(VERSION 3.7...3.20)

project (CloudRendering)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)
#MESSAGE(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.c src/*.hpp src/*.h Data/Shaders/*.glsl)
file(GLOB_RECURSE PYTORCH_MODULE_SOURCES src/PyTorch/*.cpp src/PyTorch/*.c src/PyTorch/*.hpp src/PyTorch/*.h)
file(GLOB_RECURSE PYTORCH_MODULE_SOURCES_CUDA src/PyTorch/*.cu src/PyTorch/*.cuh)
include_directories(src)

foreach(SOURCE_FILE ${PYTORCH_MODULE_SOURCES})
    list(REMOVE_ITEM SOURCES ${SOURCE_FILE})
endforeach()

file(GLOB_RECURSE SHADER_FILES Data/Shaders/*.glsl)
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/Data/Shaders" PREFIX "Shaders" FILES ${SHADER_FILES})

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    # Avoid warning 'IMPORTED_IMPLIB not set for imported target "sgl" configuration "MinSizeRel/RelWithDebInfo".'
    set(CMAKE_CONFIGURATION_TYPES Debug Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W3 /EHsc /Zc:__cplusplus")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif()

option(USE_GTEST "Build the GoogleTest test suites." OFF)
option(BUILD_PYTORCH_MODULE "Build the PyTorch module." OFF)

set(DATA_PATH "${CMAKE_SOURCE_DIR}/Data" CACHE PATH "Location of folder 'Data'")
add_definitions(-DDATA_PATH=\"${DATA_PATH}\")

set(
        VPT_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/CloudData.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/MomentUtils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PathTracer/VolumetricPathTracingPass.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PathTracer/SuperVoxelGrid.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/PathTracer/OpenExrLoader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Denoiser/Denoiser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Denoiser/EAWDenoiser.cpp
)

if (${USE_GTEST})
    set(
            GTEST_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/test/VolumetricPathTracingTestData.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test/VolumetricPathTracingTestRenderer.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/test/TestVolumetricPathTracing.cpp
    )
endif()

if (${BUILD_PYTORCH_MODULE})
    find_package(Torch QUIET)
endif()
if (${BUILD_PYTORCH_MODULE} AND NOT Torch_FOUND)
    MESSAGE(STATUS "The PyTorch module build was enabled, but PyTorch couldn't be found. Disabling PyTorch module.")
endif()


if ((UNIX OR MSYS) AND (NOT APPLE OR NOT VCPKG_TOOLCHAIN))
    find_package(PkgConfig REQUIRED)
endif()

set(FPHSA_NAME_MISMATCHED TRUE)
if(VCPKG_TOOLCHAIN)
    find_package(SDL2 CONFIG REQUIRED)
else()
    find_package(SDL2 REQUIRED)
endif()

find_package(sgl REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
set(FPHSA_NAME_MISMATCHED TRUE)
if(VCPKG_TOOLCHAIN)
    find_package(tinyxml2 CONFIG REQUIRED)
    find_package(jsoncpp CONFIG REQUIRED)
else()
    find_package(TinyXML2 REQUIRED)
    find_package(jsoncpp QUIET)
    if (NOT jsoncpp_FOUND)
        pkg_check_modules(JSONCPP jsoncpp)
    endif()
endif()

get_target_property(SGL_INTERFACE_COMPILE_DEFINITIONS sgl INTERFACE_COMPILE_DEFINITIONS)
if ("SUPPORT_VULKAN" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
	set(SUPPORT_VULKAN ON)
endif()

if ("SUPPORT_CUDA_INTEROP" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
    if (NOT OptiX_INSTALL_DIR)
        if (WIN32 AND NOT OptiX_INSTALL_DIR)
            file(GLOB OPTIX_DIR_LIST "C:/ProgramData/NVIDIA Corporation/OptiX*")
            if (OPTIX_DIR_LIST)
                list(LENGTH OPTIX_DIR_LIST OPTIX_DIR_LIST_LEN)
                math(EXPR OPTIX_DIR_LIST_LAST "${OPTIX_DIR_LIST_LEN} - 1")
                list(GET OPTIX_DIR_LIST ${OPTIX_DIR_LIST_LAST} OPTIX_DIR)
            endif()
        else()
            file(GLOB OPTIX_DIR_LIST "$ENV{HOME}/nvidia/NVIDIA-OptiX-SDK*")
            if (OPTIX_DIR_LIST)
                list(LENGTH OPTIX_DIR_LIST OPTIX_DIR_LIST_LEN)
                math(EXPR OPTIX_DIR_LIST_LAST "${OPTIX_DIR_LIST_LEN} - 1")
                list(GET OPTIX_DIR_LIST ${OPTIX_DIR_LIST_LAST} OPTIX_DIR)
            endif()
        endif()
        if (OPTIX_DIR)
            set(OptiX_INSTALL_DIR "${OPTIX_DIR}" CACHE PATH "Path to OptiX installation.")
        endif()
    endif()
    if (OptiX_INSTALL_DIR)
        list(APPEND CMAKE_MODULE_PATH ${OptiX_INSTALL_DIR}/SDK/CMake)
    endif()
    find_package(OptiX QUIET)
    if (OptiX_FOUND OR OptiX_INCLUDE)
        set(SUPPORT_OPTIX ON)
        MESSAGE(STATUS "NVIDIA OptiX found. Enabling OptiX denoiser support.")
        MESSAGE(STATUS "OptiX_INSTALL_DIR: " ${OptiX_INSTALL_DIR})
    else()
        MESSAGE(STATUS "OptiX couldn't be found. Disabling OptiX denoiser support.")
    endif()
else()
    MESSAGE(STATUS "sgl was not built with CUDA interoperability support. Disabling OptiX denoiser support.")
endif()
if (NOT SUPPORT_OPTIX)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Denoiser/OptixVptDenoiser.hpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Denoiser/OptixVptDenoiser.cpp)
endif()

find_package(Imath CONFIG QUIET)
if (NOT TARGET Imath::Imath)
    find_package(IlmBase CONFIG QUIET)
endif()
find_package(OpenEXR CONFIG QUIET)
if (NOT OpenEXR_FOUND AND NOT Imath_FOUND AND UNIX AND NOT APPLE)
    set(USE_OPENEXR_PKGCONFIG TRUE)
    pkg_check_modules(OpenEXR OpenEXR QUIET)
    pkg_check_modules(IlmBase IlmBase QUIET)
else()
    set(USE_OPENEXR_PKGCONFIG FALSE)
endif()
if (OpenEXR_FOUND AND (Imath_FOUND OR IlmBase_FOUND))
    if (${OpenEXR_VERSION} VERSION_LESS "2.3.0")
        MESSAGE(STATUS "Found OpenEXR version ${OpenEXR_VERSION} is too old. Please upgrade to version >= 2.3.0.")
    else()
        MESSAGE(STATUS "OpenEXR ${OpenEXR_VERSION} found. Enabling OpenEXR image loading support.")
        set(SUPPORT_OPENEXR true)
    endif()
else()
    MESSAGE(STATUS "OpenEXR not found. Disabling OpenEXR image loading support.")
endif()
if (NOT SUPPORT_OPENEXR)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/PathTracer/OpenExrLoader.hpp)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/PathTracer/OpenExrLoader.cpp)
    list(REMOVE_ITEM VPT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/PathTracer/OpenExrLoader.hpp)
    list(REMOVE_ITEM VPT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/PathTracer/OpenExrLoader.cpp)
endif()


if(WIN32)
    add_executable(CloudRendering WIN32 ${SOURCES})
else()
    add_executable(CloudRendering ${SOURCES})
endif()

if (APPLE)
    set_target_properties(${PROJECT}
            PROPERTIES
            OUTPUT_NAME "CloudRendering"
            MACOSX_BUNDLE_EXECUTABLE_NAME "CloudRendering"
            MACOSX_BUNDLE_BUNDLE_NAME "CloudRendering"
            FOLDER "CloudRendering"
            MACOSX_BUNDLE_SHORT_VERSION_STRING ""
            MACOSX_BUNDLE_LONG_VERSION_STRING ""
            MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/CMake/MacOSXBundleInfo.plist.in")
endif()

if (${USE_GTEST})
    include(GoogleTest)
    enable_testing()
    add_executable(CloudRendering_test ${GTEST_SOURCES} ${VPT_SOURCES})
    target_link_libraries(CloudRendering_test PRIVATE gtest gtest_main)
    target_link_libraries(CloudRendering_test PRIVATE sgl ${Boost_LIBRARIES})
    target_include_directories(CloudRendering_test PRIVATE ${sgl_INCLUDES} ${Boost_INCLUDE_DIR})
    gtest_add_tests(TARGET CloudRendering_test)
endif()

if (${BUILD_PYTORCH_MODULE} AND Torch_FOUND)
    set(PYTORCH_MODULE_ENABLED TRUE)
    if ("SUPPORT_CUDA_INTEROP" IN_LIST SGL_INTERFACE_COMPILE_DEFINITIONS)
        if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
            # https://cmake.org/cmake/help/git-stage/module/FindCUDA.html
            find_package(CUDA QUIET)
            add_library(vpt SHARED ${PYTORCH_MODULE_SOURCES} ${PYTORCH_MODULE_SOURCES_CUDA} ${VPT_SOURCES} ${SHADER_FILES})
        else()
            # https://cmake.org/cmake/help/git-stage/module/FindCUDAToolkit.html
            find_package(CUDAToolkit QUIET)
        endif()
    else()
        MESSAGE(STATUS "sgl was not built with CUDA interoperability support. Disabling PyTorch module CUDA support.")
    endif()

    if(${CUDAToolkit_FOUND} OR ${CUDA_FOUND})
        add_library(vpt SHARED ${PYTORCH_MODULE_SOURCES} ${PYTORCH_MODULE_SOURCES_CUDA} ${VPT_SOURCES} ${SHADER_FILES})
        enable_language(CUDA)
        set_target_properties(vpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_compile_definitions(vpt PRIVATE SUPPORT_CUDA_INTEROP)
    else()
        add_library(vpt SHARED ${PYTORCH_MODULE_SOURCES} ${VPT_SOURCES} ${SHADER_FILES})
    endif()

    if(${CUDAToolkit_FOUND})
        # https://en.wikipedia.org/wiki/CUDA
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability
        if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.1")
            set_property(TARGET vpt PROPERTY CUDA_ARCHITECTURES 61 75 86)
        elseif (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "10.0")
            set_property(TARGET vpt PROPERTY CUDA_ARCHITECTURES 61 75)
        else()
            set_property(TARGET vpt PROPERTY CUDA_ARCHITECTURES 61)
        endif()
        set_target_properties(vpt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        MESSAGE(STATUS "CUDA toolkit version: " ${CUDAToolkit_VERSION})
    endif()

    target_link_libraries(vpt PRIVATE ${TORCH_LIBRARIES})
    target_include_directories(vpt PRIVATE ${TORCH_INCLUDE_DIR})
    target_link_libraries(vpt PRIVATE sgl ${Boost_LIBRARIES})
    target_include_directories(vpt PRIVATE ${sgl_INCLUDES} ${Boost_INCLUDE_DIR})
    install(
            TARGETS vpt
            ARCHIVE DESTINATION modules
            LIBRARY DESTINATION modules
            RUNTIME DESTINATION bin
    )
endif()


# Hide console window in release build on Windows/MSVC.
# See: https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/recipe/VSConfigSpecificSettings
if(MSVC)
    set_target_properties(CloudRendering PROPERTIES LINK_FLAGS_DEBUG "/SUBSYSTEM:CONSOLE")
    set_target_properties(CloudRendering PROPERTIES COMPILE_DEFINITIONS_DEBUG "_CONSOLE")
    set_target_properties(CloudRendering PROPERTIES LINK_FLAGS_RELWITHDEBINFO "/SUBSYSTEM:CONSOLE")
    set_target_properties(CloudRendering PROPERTIES COMPILE_DEFINITIONS_RELWITHDEBINFO "_CONSOLE")
    set_target_properties(CloudRendering PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:WINDOWS")
    set_target_properties(CloudRendering PROPERTIES LINK_FLAGS_MINSIZEREL "/SUBSYSTEM:WINDOWS")
endif()

if(MSVC)
    #target_compile_definitions(CloudRendering PRIVATE /wd4251)
    set_target_properties(CloudRendering PROPERTIES COMPILE_FLAGS "/wd4251")
endif()

if (${SUPPORT_OPTIX})
    set(CMAKE_MODULE_PATH ${OptiX_INSTALL_DIR}/SDK/CMake ${CMAKE_MODULE_PATH})
    find_package(OptiX QUIET)
    if (OptiX_FOUND OR OptiX_INCLUDE)
        target_compile_definitions(CloudRendering PRIVATE SUPPORT_OPTIX)
        target_include_directories(CloudRendering PRIVATE ${OptiX_INCLUDE})
        if(UNIX AND NOT APPLE)
            target_link_libraries(CloudRendering PRIVATE dl)
        endif()
		if (MSYS)
		    # GetProcAddress in optix_stubs.h performs no function pointer conversion.
		    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
			# Compensate for "#pragma comment(lib, "Cfgmgr32.lib")" not working with MinGW.
            target_link_libraries(LineVis PRIVATE cfgmgr32)
		endif()
    endif()
endif()

if((${CMAKE_GENERATOR} STREQUAL "MinGW Makefiles") OR (${CMAKE_GENERATOR} STREQUAL "MSYS Makefiles"))
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mwindows")
    target_link_libraries(CloudRendering PUBLIC mingw32)
endif()
if(VCPKG_TOOLCHAIN)
    target_link_libraries(CloudRendering PUBLIC SDL2::SDL2main)
else()
    target_link_libraries(CloudRendering PUBLIC SDL2::Main)
endif()

# sgl provides Vulkan headers in its interface, which are provided if Vulkan was not found during its build.
find_package(Vulkan QUIET)
if (${Vulkan_FOUND} AND ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.21.0")
    target_link_libraries(CloudRendering PUBLIC Vulkan::Headers)
endif()

target_link_libraries(CloudRendering PUBLIC sgl ${Boost_LIBRARIES})
include_directories(${sgl_INCLUDES} ${Boost_INCLUDE_DIR})

if(VCPKG_TOOLCHAIN)
    target_link_libraries(CloudRendering PRIVATE tinyxml2::tinyxml2)
else()
    target_link_libraries(CloudRendering PRIVATE ${TINYXML2_LIBRARIES})
endif()
target_include_directories(CloudRendering PRIVATE ${sgl_INCLUDES} ${TINYXML2_INCLUDE_DIR})

if (JSONCPP_LIBRARIES)
    target_link_libraries(CloudRendering PRIVATE ${JSONCPP_LIBRARIES})
    target_link_directories(CloudRendering PRIVATE ${JSONCPP_LIBRARY_DIRS})
    target_include_directories(CloudRendering PRIVATE ${JSONCPP_INCLUDE_DIRS})
    if (${USE_GTEST})
        target_link_libraries(CloudRendering_test PRIVATE ${JSONCPP_LIBRARIES})
        target_link_directories(CloudRendering_test PRIVATE ${JSONCPP_LIBRARY_DIRS})
        target_include_directories(CloudRendering_test PRIVATE ${JSONCPP_INCLUDE_DIRS})
    endif()
    if (${PYTORCH_MODULE_ENABLED})
        target_link_libraries(vpt PRIVATE ${JSONCPP_LIBRARIES})
        target_link_directories(vpt PRIVATE ${JSONCPP_LIBRARY_DIRS})
        target_include_directories(vpt PRIVATE ${JSONCPP_INCLUDE_DIRS})
    endif()
endif()
if (TARGET jsoncpp_lib)
    get_target_property(jsoncpp_INCLUDES jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
    target_link_libraries(CloudRendering PRIVATE jsoncpp_lib)
    if (${USE_GTEST})
        target_link_libraries(CloudRendering_test PRIVATE jsoncpp_lib)
    endif()
    if (${PYTORCH_MODULE_ENABLED})
        target_link_libraries(vpt PRIVATE jsoncpp_lib)
    endif()
elseif(TARGET jsoncpp_static)
    get_target_property(jsoncpp_INCLUDES jsoncpp_static INTERFACE_INCLUDE_DIRECTORIES)
    target_link_libraries(CloudRendering PRIVATE jsoncpp_static)
    if (${USE_GTEST})
        target_link_libraries(CloudRendering_test PRIVATE jsoncpp_static)
    endif()
    if (${PYTORCH_MODULE_ENABLED})
        target_link_libraries(vpt PRIVATE jsoncpp_static)
    endif()
endif()
if (TARGET jsoncpp_object)
    target_link_libraries(CloudRendering PRIVATE jsoncpp_object)
    if (${USE_GTEST})
        target_link_libraries(CloudRendering_test PRIVATE jsoncpp_object)
    endif()
    if (${PYTORCH_MODULE_ENABLED})
        target_link_libraries(vpt PRIVATE jsoncpp_object)
    endif()
endif()
target_include_directories(CloudRendering PRIVATE ${jsoncpp_INCLUDES})
if (${USE_GTEST})
    target_include_directories(CloudRendering_test PRIVATE ${jsoncpp_INCLUDES})
endif()
if (${PYTORCH_MODULE_ENABLED})
    target_include_directories(vpt PRIVATE ${jsoncpp_INCLUDES})
endif()

if (SUPPORT_OPENEXR)
    if (${USE_OPENEXR_PKGCONFIG})
        if (OpenEXR_FOUND)
            target_link_libraries(CloudRendering PRIVATE ${OpenEXR_LIBRARIES})
            if (${USE_GTEST})
                target_link_libraries(CloudRendering_test PRIVATE ${OpenEXR_LIBRARIES})
            endif()
            if (${PYTORCH_MODULE_ENABLED})
                target_link_libraries(vpt PRIVATE ${OpenEXR_LIBRARIES})
            endif()
        endif()
        if (IlmBase_FOUND)
            target_link_libraries(CloudRendering PRIVATE ${IlmBase_LIBRARIES})
            if (${USE_GTEST})
                target_link_libraries(CloudRendering_test PRIVATE ${IlmBase_LIBRARIES})
            endif()
            if (${PYTORCH_MODULE_ENABLED})
                target_link_libraries(vpt PRIVATE ${IlmBase_LIBRARIES})
            endif()
        endif()
    else()
        target_link_libraries(
                CloudRendering PRIVATE
                # For OpenEXR/Imath 3.x:
                $<$<TARGET_EXISTS:OpenEXR::OpenEXR>:OpenEXR::OpenEXR>
                $<$<TARGET_EXISTS:Imath::Imath>:Imath::Imath>
                $<$<TARGET_EXISTS:Imath::Half>:Imath::Half>
                # For OpenEXR 2.4/2.5:
                $<$<TARGET_EXISTS:OpenEXR::IlmImf>:OpenEXR::IlmImf>
                $<$<TARGET_EXISTS:IlmBase::Imath>:IlmBase::Imath>
                $<$<TARGET_EXISTS:IlmBase::Half>:IlmBase::Half>
                $<$<TARGET_EXISTS:IlmBase::IlmThread>:IlmBase::IlmThread>
                $<$<TARGET_EXISTS:IlmBase::Iex>:IlmBase::Iex>
        )
        if (${USE_GTEST})
            target_link_libraries(
                    CloudRendering_test PRIVATE
                    # For OpenEXR/Imath 3.x:
                    $<$<TARGET_EXISTS:OpenEXR::OpenEXR>:OpenEXR::OpenEXR>
                    $<$<TARGET_EXISTS:Imath::Imath>:Imath::Imath>
                    $<$<TARGET_EXISTS:Imath::Half>:Imath::Half>
                    # For OpenEXR 2.4/2.5:
                    $<$<TARGET_EXISTS:OpenEXR::IlmImf>:OpenEXR::IlmImf>
                    $<$<TARGET_EXISTS:IlmBase::Imath>:IlmBase::Imath>
                    $<$<TARGET_EXISTS:IlmBase::Half>:IlmBase::Half>
                    $<$<TARGET_EXISTS:IlmBase::IlmThread>:IlmBase::IlmThread>
                    $<$<TARGET_EXISTS:IlmBase::Iex>:IlmBase::Iex>
            )
        endif()
        if (${PYTORCH_MODULE_ENABLED})
            target_link_libraries(
                    vpt PRIVATE
                    # For OpenEXR/Imath 3.x:
                    $<$<TARGET_EXISTS:OpenEXR::OpenEXR>:OpenEXR::OpenEXR>
                    $<$<TARGET_EXISTS:Imath::Imath>:Imath::Imath>
                    $<$<TARGET_EXISTS:Imath::Half>:Imath::Half>
                    # For OpenEXR 2.4/2.5:
                    $<$<TARGET_EXISTS:OpenEXR::IlmImf>:OpenEXR::IlmImf>
                    $<$<TARGET_EXISTS:IlmBase::Imath>:IlmBase::Imath>
                    $<$<TARGET_EXISTS:IlmBase::Half>:IlmBase::Half>
                    $<$<TARGET_EXISTS:IlmBase::IlmThread>:IlmBase::IlmThread>
                    $<$<TARGET_EXISTS:IlmBase::Iex>:IlmBase::Iex>
            )
        endif()
    endif()
    target_compile_definitions(CloudRendering PRIVATE SUPPORT_OPENEXR)
    if (${USE_GTEST})
        target_compile_definitions(CloudRendering_test PRIVATE SUPPORT_OPENEXR)
    endif()
    if (${PYTORCH_MODULE_ENABLED})
        target_compile_definitions(vpt PRIVATE SUPPORT_OPENEXR)
    endif()
endif()


# According to https://devblogs.microsoft.com/cppblog/improved-openmp-support-for-cpp-in-visual-studio/,
# support for LLVM OpenMP was added with Visual Studio 2019 version 16.9. According to
# https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macros?view=msvc-170,
# both Visual Studio 2019 version 16.8 and 16.9 have _MSC_VER / MSVC_VERSION 1928, so we need to check
# for a minimum version 1929.
find_package(OpenMP REQUIRED)
if(MSVC)
    option(USE_MSVC_OPENMP_LLVM "Use LLVM OpenMP with MSVC (available since Visual Studio 2019 version 16.9)." ON)
    option(USE_MSVC_OPENMP_EXPERIMENTAL "Use experimental OpenMP with MSVC (available since Visual Studio 2019 version 16.9)." ON)

    if(${MSVC_VERSION} GREATER_EQUAL 1929 AND (USE_MSVC_OPENMP_LLVM OR USE_MSVC_OPENMP_EXPERIMENTAL))
        if(USE_MSVC_OPENMP_LLVM)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp:llvm")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:llvm")
        endif()
        if(USE_MSVC_OPENMP_EXPERIMENTAL)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp:experimental")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp:experimental")
        endif()
        if(USE_MSVC_OPENMP_LLVM AND USE_MSVC_OPENMP_EXPERIMENTAL)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w44005")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w44005")
        endif()
    endif()
else()
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

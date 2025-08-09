# BSD 2-Clause License
#
# Copyright (c) 2024, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
import os
import platform
import glob
import shutil
import subprocess
import urllib
import zipfile
import tarfile
from pathlib import Path
from urllib.request import urlopen
import setuptools
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.egg_info import egg_info
from setuptools.dist import Distribution
from setuptools.command import bdist_egg
try:
    from tkinter import messagebox
except ImportError:
    messagebox = None


uses_pip = \
    ('_' in os.environ and (os.environ['_'].endswith('pip') or os.environ['_'].endswith('pip3'))) \
    or 'PIP_BUILD_TRACKER' in os.environ

if os.name != 'nt':
    # torch.utils.cpp_extension.BuildExtension unfortunately has no support for C sources.
    # We use our own compiler wrapper and check whether the pattern "-c <source>.c" appears.
    # On Windows, we could use "/Tc" and "/Tp" to force the use of C or C++, respectively
    # (https://stackoverflow.com/questions/32505294/c-vs-c-using-cl-compiler-in-vs2015).
    cxx_compiler = os.environ.get('CXX', None)
    cc_compiler = os.environ.get('CC', None)
    if cxx_compiler is not None:
        os.environ['CXX_REAL'] = cxx_compiler
    if cc_compiler is not None:
        os.environ['CC_REAL'] = cc_compiler
    os.environ['CXX'] = os.path.abspath('third_party/custom/compiler.sh')
    os.environ['CC'] = os.path.abspath('third_party/custom/compiler.sh')

from torch.utils.cpp_extension import include_paths, library_paths, BuildExtension, CUDAExtension, IS_WINDOWS, IS_HIP_EXTENSION, ROCM_VERSION

extra_compile_args_cxx = []
extra_compile_args_nvcc = []
extra_compile_args = []
extra_link_args = []
if IS_WINDOWS:
    extra_compile_args_cxx.append('/std:c++17')
    extra_compile_args_cxx.append('/Zc:__cplusplus')
    extra_compile_args_nvcc.append('-std=c++17')
    extra_compile_args_cxx.append('/openmp')
else:
    extra_compile_args.append('-std=c++17')
    extra_compile_args_cxx.append('-fopenmp')
    extra_link_args.append('-Wl,-rpath=\'$ORIGIN\'')


class EggInfoInstallLicense(egg_info):
    def run(self):
        if not self.distribution.have_run.get('install', True):
            self.mkpath(self.egg_info)
            self.copy_file('LICENSE', self.egg_info)
        egg_info.run(self)


def find_all_sources_in_dir(root_dir, blacklist=None, no_recurse=False):
    source_files = []
    for root, subdirs, files in os.walk(root_dir):
        for filename in files:
            if blacklist is not None and filename in blacklist:
                continue
            if filename.endswith('.cpp') or filename.endswith('.cc') or filename.endswith('.c') or filename.endswith('.cu'):
                source_files.append(root + "/" + filename)
        if no_recurse:
            break
    return source_files


def find_all_source_names_in_dir(root_dir):
    source_files = []
    for root, subdirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.cpp') or filename.endswith('.cc') or filename.endswith('.c') or filename.endswith('.cu'):
                source_files.append(filename)
    return source_files


def find_all_shared_libs_in_dir(root_dir, prefix=None):
    shared_lib_files = []
    for root, subdirs, files in os.walk(root_dir):
        for filename in files:
            if prefix is not None and not filename.startswith(prefix):
                continue
            if filename.endswith('.dll') or filename.endswith('.so') or '.so.' in filename:
                shared_lib_files.append(root + "/" + filename)
    return shared_lib_files


def get_cmake_exec():
    cmake_exec = 'cmake'
    if IS_WINDOWS:
        # CMake on Windows is usually not in the PATH. If it is not found, try to use the default location.
        cmake_exec = shutil.which('cmake')
        cmake_default_path = 'C:\\Program Files\\CMake\\bin\\cmake.exe'
        if cmake_exec is None and os.path.isfile(cmake_default_path):
            cmake_exec = cmake_default_path
    return cmake_exec


# Fetch all dependencies.
if not os.path.exists('third_party/sgl'):
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/chrismile/sgl.git', 'third_party/sgl'])
if not os.path.exists('third_party/glm-src'):
    subprocess.run(['git', 'clone', 'https://github.com/g-truc/glm.git', 'third_party/glm-src'])
if not os.path.exists('third_party/jsoncpp-src'):
    subprocess.run(['git', 'clone', 'https://github.com/open-source-parsers/jsoncpp.git', 'third_party/jsoncpp-src'])
if not os.path.exists('third_party/glslang-src'):
    subprocess.run(['git', 'clone', 'https://github.com/KhronosGroup/glslang.git', 'third_party/glslang-src'])
    subprocess.run(['git', '-C', 'third_party/glslang-src', 'checkout', 'f754c852a87988eb097a39480c65f704ceb46274'])
    Path('third_party/glslang').mkdir(parents=True, exist_ok=True)
    shutil.copytree('third_party/glslang-src/SPIRV', 'third_party/glslang/SPIRV')
if not os.path.exists('third_party/tinyxml2-src'):
    subprocess.run(['git', 'clone', 'https://github.com/leethomason/tinyxml2.git', 'third_party/tinyxml2-src'])
if not os.path.exists('third_party/Imath-src'):
    subprocess.run(['git', 'clone', 'https://github.com/AcademySoftwareFoundation/Imath.git', 'third_party/Imath-src'])
    subprocess.run(['git', '-C', 'third_party/Imath-src', 'checkout', 'aa28eb56e2be4547e220f2dc7bbf9961e2d8c9c4'])
if not os.path.exists('third_party/openexr-src'):
    subprocess.run(['git', 'clone', 'https://github.com/AcademySoftwareFoundation/openexr.git', 'third_party/openexr-src'])
    subprocess.run(['git', '-C', 'third_party/openexr-src', 'checkout', 'ad85f1a2677f92e381da1bdbd390eb6b2da4d7ae'])
if not os.path.exists('third_party/libdeflate-src'):
    subprocess.run(['git', 'clone', 'https://github.com/ebiggers/libdeflate.git', 'third_party/libdeflate-src'])
    subprocess.run(['git', '-C', 'third_party/libdeflate-src', 'checkout', '78051988f96dc8d8916310d8b24021f01bd9e102'])
if not os.path.exists('third_party/openvdb-src'):
    subprocess.run(['git', 'clone', 'https://github.com/AcademySoftwareFoundation/openvdb.git', 'third_party/openvdb-src'])
    subprocess.run(['git', '-C', 'third_party/openvdb-src', 'checkout', 'a4705352e0e3ecb1f82eff2eca0c1b061ab7656b'])
if not os.path.exists('third_party/libpng-src'):
    subprocess.run(['git', 'clone', 'https://github.com/pnggroup/libpng.git', 'third_party/libpng-src'])
if not os.path.exists('third_party/boost-src'):
    subprocess.run(['git', 'clone', 'https://github.com/boostorg/boost.git', 'third_party/boost-src'])
    subprocess.run(['git', '-C', 'third_party/boost-src', 'checkout', 'boost-1.87.0'])
    subprocess.run(['git', '-C', 'third_party/boost-src', 'submodule', 'update', '--init', '--recursive'])
if not os.path.exists('third_party/tbb-src'):
    subprocess.run(['git', 'clone', 'https://github.com/uxlfoundation/oneTBB.git', 'third_party/tbb-src'])
    subprocess.run(['git', '-C', 'third_party/tbb-src', 'checkout', '8659619fbd73361a129e02df19d24aa5054649cb'])
if not os.path.exists('third_party/blosc-src'):
    subprocess.run(['git', 'clone', 'https://github.com/Blosc/c-blosc.git', 'third_party/blosc-src'])
    subprocess.run(['git', '-C', 'third_party/blosc-src', 'checkout', 'dcf6813d46f9fcbf7af4395b5536f886dfc76dd0'])
    with open('third_party/blosc-src/internal-complibs/zstd-1.5.6/decompress/huf_decompress_amd64.c', 'w') as file:
        file.write('#include "huf_decompress_amd64.S"\n')

oidn_version = '2.3.1'
if platform.machine() == 'x86_64' or platform.machine() == 'AMD64':
    os_arch = 'x86_64'
else:
    os_arch = 'aarch64'
if IS_WINDOWS:
    oidn_folder_name = f'oidn-{oidn_version}.x64.windows'
    oidn_archive_name = f'{oidn_folder_name}.zip'
else:
    oidn_folder_name = f'oidn-{oidn_version}.{os_arch}.linux'
    oidn_archive_name = f'{oidn_folder_name}.tar.gz'
oidn_url = f'https://github.com/OpenImageDenoise/oidn/releases/download/v{oidn_version}/{oidn_archive_name}'
if not os.path.isdir(f'third_party/{oidn_folder_name}'):
    urllib.request.urlretrieve(oidn_url, f'third_party/{oidn_archive_name}')
    if IS_WINDOWS:
        with zipfile.ZipFile(f'third_party/{oidn_archive_name}', 'r') as zip_ref:
            zip_ref.extractall(f'third_party')
    else:
        with tarfile.open(f'third_party/{oidn_archive_name}', 'r') as zip_ref:
            zip_ref.extractall(f'third_party')

if not os.path.exists('submodules/IsosurfaceCpp/src'):
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive'])

include_dirs = [
    'src',
    'modules/kpn_module/src',
    'third_party',
    'third_party/sgl/src',
    'third_party/sgl/src/Graphics/Vulkan/libs',
    'third_party/sgl/src/Graphics/Vulkan/libs/Vulkan-Headers',
    'third_party/glm-src',
    'third_party/tinyxml2-src',
    'third_party/jsoncpp-src/include',
    'third_party/custom',
    'third_party/custom/libpng',
    'third_party/custom/openvdb',
    'third_party/custom/OpenEXR',
    'third_party/custom/Imath',
    'third_party/glslang-src',
    'third_party/Imath-src/src',
    'third_party/Imath-src/src/Imath',
    'third_party/openexr-src/src/lib',
    'third_party/openexr-src/src/lib/Iex',
    'third_party/openexr-src/src/lib/IlmThread',
    'third_party/openexr-src/src/lib/OpenEXR',
    'third_party/openexr-src/src/lib/OpenEXRCore',
    'third_party/openexr-src/src/lib/OpenEXRUtil',
    'third_party/libdeflate-src',
    'third_party/libdeflate-src/lib',
    'third_party/openvdb-src/openvdb',
    'third_party/libpng-src',
    'third_party/tbb-src/include',
    'third_party/blosc-src/blosc',
    'third_party/blosc-src/internal-complibs/lz4-1.10.0',
    'third_party/blosc-src/internal-complibs/zlib-1.3.1',
    'third_party/blosc-src/internal-complibs/zstd-1.5.6',
    'submodules/IsosurfaceCpp/src',
    'submodules',
    'third_party/boost-src/libs/algorithm/include',
    'third_party/boost-src/libs/assert/include',  # dependency by intrusive
    'third_party/boost-src/libs/config/include',
    'third_party/boost-src/libs/concept_check/include',  # dependency by range
    'third_party/boost-src/libs/container/include',  # dependency by interprocess (Windows only maybe)
    'third_party/boost-src/libs/core/include',
    'third_party/boost-src/libs/detail/include',  # dependency by interprocess (Windows only maybe)
    'third_party/boost-src/libs/exception/include',
    'third_party/boost-src/libs/integer/include',
    'third_party/boost-src/libs/interprocess/include',
    'third_party/boost-src/libs/intrusive/include',  # dependency by container
    'third_party/boost-src/libs/iostreams/include',
    'third_party/boost-src/libs/iterator/include',  # dependency by range
    'third_party/boost-src/libs/move/include',  # dependency by container
    'third_party/boost-src/libs/mpl/include',  # dependency by numeric
    'third_party/boost-src/libs/numeric/conversion/include',
    'third_party/boost-src/libs/predef/include',  # dependency by winapi
    'third_party/boost-src/libs/preprocessor/include',  # dependency by numeric
    'third_party/boost-src/libs/range/include',  # dependency by iostreams
    'third_party/boost-src/libs/smart_ptr/include',  # dependency by iostreams
    'third_party/boost-src/libs/static_assert/include',  # dependency by iostreams
    'third_party/boost-src/libs/throw_exception/include',
    'third_party/boost-src/libs/type_traits/include',  # dependency by numeric
    'third_party/boost-src/libs/utility/include',  # dependency by range
]
if IS_WINDOWS:
    # Dependency by interprocess
    include_dirs.append('third_party/boost-src/libs/winapi/include')
local_oidn_include_path = f'third_party/{oidn_folder_name}/include'
if IS_WINDOWS:
    # nvcc has issues with the dots in the path...
    #extra_compile_args_cxx.append(f'/I "{os.path.abspath(local_oidn_include_path)}"')
    extra_compile_args_cxx.append('/I')
    extra_compile_args_cxx.append(f'{os.path.abspath(local_oidn_include_path)}')
else:
    include_dirs.append(local_oidn_include_path)
source_files = []
src_blacklist = {'Main.cpp', 'MainApp.cpp', 'MainAppState.cpp', 'DataView.cpp'}
source_files += find_all_sources_in_dir('src', src_blacklist)
source_files += find_all_sources_in_dir('modules/kpn_module/src')
source_files += [
    'third_party/sgl/src/Math/Geometry/MatrixUtil.cpp',
    'third_party/sgl/src/Math/Geometry/Plane.cpp',
    'third_party/sgl/src/Math/Geometry/AABB2.cpp',
    'third_party/sgl/src/Math/Geometry/AABB3.cpp',
    'third_party/sgl/src/Math/Geometry/Ray3.cpp',
    'third_party/sgl/src/Utils/Dialog.cpp',
    'third_party/sgl/src/Utils/StringUtils.cpp',
    'third_party/sgl/src/Utils/Env.cpp',
    'third_party/sgl/src/Utils/Convert.cpp',
    'third_party/sgl/src/Utils/AppSettings.cpp',
    'third_party/sgl/src/Utils/Timer.cpp',
    'third_party/sgl/src/Utils/XML.cpp',
    'third_party/sgl/src/Utils/File/Logfile.cpp',
    'third_party/sgl/src/Utils/File/FileUtils.cpp',
    'third_party/sgl/src/Utils/File/Execute.cpp',
    'third_party/sgl/src/Utils/File/FileLoader.cpp',
    'third_party/sgl/src/Utils/File/LineReader.cpp',
    'third_party/sgl/src/Utils/File/CsvParser.cpp',
    'third_party/sgl/src/Utils/File/PathWatch.cpp',
    'third_party/sgl/src/Utils/Events/EventManager.cpp',
    'third_party/sgl/src/Utils/Events/Stream/BinaryStream.cpp',
    'third_party/sgl/src/Utils/Events/Stream/StringStream.cpp',
    'third_party/sgl/src/Utils/Json/SimpleJson.cpp',
    'third_party/sgl/src/Utils/Regex/Tokens.cpp',
    'third_party/sgl/src/Utils/Regex/TransformString.cpp',
    'third_party/sgl/src/Utils/Parallel/Histogram.cpp',
    'third_party/sgl/src/Utils/Parallel/Reduction.cpp',
    'third_party/sgl/src/Utils/Mesh/IndexMesh.cpp',
    'third_party/sgl/src/Graphics/Color.cpp',
    'third_party/sgl/src/Graphics/Texture/Bitmap.cpp',
    'third_party/sgl/src/Graphics/Scene/Camera.cpp',
    'third_party/sgl/src/Graphics/Scene/CameraHelper.cpp',
    'third_party/sgl/src/Graphics/Scene/RenderTarget.cpp',
    'third_party/sgl/src/Graphics/Compression/Compression.cpp',
    'third_party/sgl/src/Graphics/GLSL/PreprocessorGlsl.cpp',
    'third_party/sgl/src/Graphics/Vulkan/libs/volk/volk.c',
    'third_party/sgl/src/Graphics/Vulkan/libs/SPIRV-Reflect/spirv_reflect.c',
    'third_party/sgl/src/Graphics/Vulkan/Utils/VmaImpl.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Memory.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Status.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Instance.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Device.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Swapchain.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/SyncObjects.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/Timer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Utils/InteropCustom.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Buffers/Buffer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Buffers/Framebuffer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Image/Image.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Shader/Shader.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Shader/ShaderManager.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/CommandBuffer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Renderer.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Data.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Pipeline.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/ComputePipeline.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/GraphicsPipeline.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/RayTracingPipeline.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/AccelerationStructure.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Passes/Pass.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Passes/BlitRenderPass.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Passes/BlitComputePass.cpp',
    'third_party/sgl/src/Graphics/Vulkan/Render/Helpers.cpp',
    'third_party/sgl/src/ImGui/Widgets/TransferFunctionWindow.cpp',
    'third_party/sgl/src/ImGui/Widgets/MultiVarTransferFunctionWindow.cpp',
]
source_files += find_all_sources_in_dir('submodules/IsosurfaceCpp/src')
source_files += [
    'third_party/tinyxml2-src/tinyxml2.cpp',
]
source_files += find_all_sources_in_dir('third_party/jsoncpp-src/src/lib_json')
source_files += find_all_sources_in_dir('third_party/glslang-src/SPIRV')
source_files += find_all_sources_in_dir('third_party/glslang-src/glslang/CInterface')
source_files += find_all_sources_in_dir('third_party/glslang-src/glslang/GenericCodeGen')
source_files += find_all_sources_in_dir('third_party/glslang-src/glslang/MachineIndependent')
source_files += find_all_sources_in_dir('third_party/glslang-src/glslang/ResourceLimits')
if IS_WINDOWS:
    source_files += find_all_sources_in_dir('third_party/glslang-src/glslang/OSDependent/Windows')
else:
    source_files += find_all_sources_in_dir('third_party/glslang-src/glslang/OSDependent/Unix')
source_files += find_all_sources_in_dir('third_party/libdeflate-src/lib', no_recurse=True)
if platform.machine() == 'x86_64' or platform.machine() == 'AMD64':
    source_files += find_all_sources_in_dir('third_party/libdeflate-src/lib/x86')
else:
    source_files += find_all_sources_in_dir('third_party/libdeflate-src/lib/arm')
openvdb_blacklist = find_all_source_names_in_dir('third_party/openvdb-src/openvdb/openvdb/python')
openvdb_blacklist += find_all_source_names_in_dir('third_party/openvdb-src/openvdb/openvdb/unittest')
source_files += find_all_sources_in_dir('third_party/openvdb-src/openvdb/openvdb', openvdb_blacklist)
source_files += find_all_sources_in_dir('third_party/tbb-src/src/tbb')
# Currently, only lz4, zlib and zstd are provided via hird_party/blosc-src/internal-complibs.
boost_iostreams_blacklist = ['bzip2.cpp', 'gzip.cpp', 'lzma.cpp']
source_files += find_all_sources_in_dir('third_party/boost-src/libs/iostreams/src', boost_iostreams_blacklist)
#blosc_blacklist = []
#if not IS_WINDOWS:
#    blosc_blacklist = ['pthread.c']
source_files += find_all_sources_in_dir('third_party/blosc-src/blosc', no_recurse=True)
source_files += find_all_sources_in_dir('third_party/blosc-src/internal-complibs/lz4-1.10.0')
source_files += find_all_sources_in_dir('third_party/blosc-src/internal-complibs/zlib-1.3.1', no_recurse=True)
source_files += find_all_sources_in_dir('third_party/blosc-src/internal-complibs/zstd-1.5.6/common')
source_files += find_all_sources_in_dir('third_party/blosc-src/internal-complibs/zstd-1.5.6/compress')
source_files += find_all_sources_in_dir('third_party/blosc-src/internal-complibs/zstd-1.5.6/decompress')
source_files += find_all_sources_in_dir('third_party/blosc-src/internal-complibs/zstd-1.5.6/dictBuilder')

data_files_all = []
data_files = ['src/PyTorch/vpt.pyi']
libraries = []
library_dirs = []
extra_objects = []
defines = [
    ('USE_GLM',),
    ('SUPPORT_VULKAN',),
    ('SUPPORT_GLSLANG_BACKEND',),
    ('SUPPORT_TINYXML2',),
    ('DISABLE_IMGUI',),
    ('DISABLE_DEVICE_SELECTION_SUPPORT',),
    ('BUILD_PYTHON_MODULE',),
    ('BUILD_PYTHON_MODULE_NEW',),
    ('CUDA_HOST_COMPILER_COMPATIBLE',),
    ('MODULE_OP_API', ''),
    ('SUPPORT_PYTORCH_DENOISER',),
    ('USE_KPN_MODULE',),
    ('SUPPORT_OPEN_IMAGE_DENOISE',),
    ('SUPPORT_OPENEXR',),
    ('USE_OPENVDB',),
    ('NANOVDB_USE_OPENVDB',),
    ('USE_TBB',),
    ('KPN_MODULE_OP_API', ''),
    # For OpenEXR.
    ('ILMTHREAD_USE_TBB',),
    # For glslang.
    ('ENABLE_SPIRV',),
    # For c-blosc.
    ('HAVE_LZ4',),
    ('HAVE_ZLIB',),
    ('HAVE_ZSTD',),
    # For OpenVDB.
    ('OPENVDB_PRIVATE',),
    ('OPENVDB_USE_DELAYED_LOADING',),
    ('OPENVDB_USE_BLOSC',),
    # ('OPENVDB_STATICLIB',),
    # For TBB.
    ('__TBB_SOURCE_DIRECTLY_INCLUDED',),
]
# Change symbol visibility?
if IS_WINDOWS:
    defines.append(('DLL_OBJECT', ''))
    defines.append(('DISABLE_SINGLETON_BOOST_INTERPROCESS',))
    # According to https://learn.microsoft.com/en-us/windows/win32/api/shlwapi/nf-shlwapi-pathremovefilespecw,
    # shlwapi.lib and shlwapi.dll both exist. Maybe this should rather be a extra_objects file?
    libraries.append('shlwapi')
    libraries.append('shell32')
    libraries.append('user32')
    # libraries.append('cfgmgr32')
    defines.append(('GLSLANG_OSINCLUDE_WIN32', ''))
else:
    defines.append(('DLL_OBJECT', ''))
    #extra_compile_args.append('-O0')  # For debugging tests.
    #extra_compile_args.append('-ggdb')  # For debugging tests.
    libraries.append('dl')
    defines.append(('GLSLANG_OSINCLUDE_UNIX', ''))


# TODO: Add support for not using CUDA.
defines.append(('SUPPORT_CUDA_INTEROP',))
defines.append(('USE_CUDA',))
source_files.append('third_party/sgl/src/Graphics/Utils/InteropCuda.cpp')
source_files.append('third_party/sgl/src/Graphics/Vulkan/Utils/InteropCuda.cpp')


libpng_blacklist = ['pngtest.c']
source_files += find_all_sources_in_dir('third_party/libpng-src', libpng_blacklist, no_recurse=True)
if platform.machine() == 'x86_64' or platform.machine() == 'AMD64':
    source_files += find_all_sources_in_dir('third_party/libpng-src/intel')
    defines.append(('PNG_INTEL_SSE_OPT', '1'))
else:
    source_files += find_all_sources_in_dir('third_party/libpng-src/arm')
    defines.append(('PNG_ARM_NEON_OPT', '2'))


imath_blacklist = ['toFloat.cpp']
source_files_openexr = find_all_sources_in_dir('third_party/Imath-src/src/Imath', imath_blacklist)
openexr_blacklist = ['b44ExpLogTable.cpp', 'dwaLookups.cpp']
source_files_openexr += find_all_sources_in_dir('third_party/openexr-src/src/lib', openexr_blacklist)
if not IS_WINDOWS:
    source_files += source_files_openexr
elif not os.path.isfile('third_party/tmp/openexr.lib'):
    # On Windows, we cannot compile all source files to one shared library due to command length limitations.
    # This is also discussed in:
    # - https://github.com/pypa/setuptools/issues/4177
    # - https://github.com/pypa/distutils/issues/226
    # "pip install ." will terminate with the message "error: command 'C:\\<path>\\link.exe' failed: None".
    # If we run link.exe manually, one receives the error message "The filename or extension is too long".
    # This is because internally Windows has the following limit for command lengths and the file paths are too long.
    # https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-createprocessa#parameters
    # "The maximum length of this string is 32,767 characters, including the Unicode terminating null character."
    # As a solution, we compile OpenEXR and its dependency Imath separately to a static library using distutils.
    # This way, we can get to less than 30,000 characters on a test system.
    # distutils has been deprecated in Python 3.12. We will try to use the protected member "setuptools._distutils"
    # as a fallback if distutils is not available (even though this is not a nice looking hack).
    try:
        import distutils.ccompiler as ccompiler
    except ImportError:
        import setuptools._distutils.ccompiler as ccompiler
    Path('third_party/tmp').mkdir(parents=True, exist_ok=True)
    # TODO: https://gist.github.com/udnaan/d549950a33fd82d13f9e6ba4aae82964 would imply we can wrap this in an object.
    compiler = ccompiler.new_compiler()
    objects = compiler.compile(
        source_files_openexr, include_dirs=include_dirs, macros=defines, extra_preargs=extra_compile_args_cxx)
    compiler.create_static_lib(objects, 'openexr', 'third_party/tmp')

if os.path.isfile('third_party/tmp/openexr.lib'):
    extra_objects.append('third_party/tmp/openexr.lib')


# Download OpenImageDenoise.
if IS_WINDOWS:
    oidn_shared_libs = find_all_shared_libs_in_dir(f'third_party/{oidn_folder_name}/bin')
    extra_objects.append(f'third_party/{oidn_folder_name}/lib/OpenImageDenoise.lib')
    extra_objects.append(f'third_party/{oidn_folder_name}/lib/OpenImageDenoise_core.lib')
else:
    # oidn_shared_libs = find_all_shared_libs_in_dir(f'third_party/{oidn_folder_name}/lib')
    oidn_shared_libs = [
        f'third_party/{oidn_folder_name}/lib/libOpenImageDenoise.so.2',
        f'third_party/{oidn_folder_name}/lib/libOpenImageDenoise_core.so.{oidn_version}',
        f'third_party/{oidn_folder_name}/lib/libOpenImageDenoise_device_cpu.so.{oidn_version}',
        f'third_party/{oidn_folder_name}/lib/libOpenImageDenoise_device_cuda.so.{oidn_version}',
        f'third_party/{oidn_folder_name}/lib/libOpenImageDenoise_device_hip.so.{oidn_version}',
        f'third_party/{oidn_folder_name}/lib/libOpenImageDenoise_device_sycl.so.{oidn_version}',
        f'third_party/{oidn_folder_name}/lib/libtbb.so.12',  # Is a symlink, but linked to by device_cpu... :(
        f'third_party/{oidn_folder_name}/lib/libsycl.so.7',  # Is a symlink, but linked to by device_sycl... :(
        f'third_party/{oidn_folder_name}/lib/libpi_level_zero.so',
    ]
    library_dirs.append(os.path.abspath(f'third_party/{oidn_folder_name}/lib'))
    libraries.append(f'OpenImageDenoise')
    library_dirs
    oidn_shared_libs += find_all_shared_libs_in_dir(f'third_party/{oidn_folder_name}/lib', prefix='libtbbbind')
data_files += [oidn_shared_libs]

# Check whether OptiX can be found somewhere.
support_optix = False
if IS_WINDOWS:
    optix_dir_pattern = 'C:/ProgramData/NVIDIA Corporation/OptiX*'
else:
    optix_dir_pattern = os.path.join(str(Path.home()), 'nvidia/NVIDIA-OptiX-SDK*')
optix_dir_candidates = glob.glob(optix_dir_pattern)
if len(optix_dir_candidates) > 0:
    support_optix = True
    optix_dir_candidates = sorted(optix_dir_candidates)
    optix_dir = optix_dir_candidates[-1]
    defines.append(('SUPPORT_OPTIX',))
    include_dirs.append(os.path.join(optix_dir, 'include'))


def prompt_yes_no(prompt):
    if uses_pip:
        if messagebox is not None:
            res = messagebox.askquestion('User Prompt', prompt)
            return res == 'yes'
        # Unfortunately, for pip, no command line input prompts can be used... As a fallback, we assume the user
        # accepted the choice, as the DLSS license is included in this project's license anyways.
        return True
    yes_no_string = input(prompt).strip()
    return yes_no_string == 'y' or yes_no_string == 'yes'


def check_dlss():
    if os.path.isdir('third_party/DLSS'):
        return True
    if not prompt_yes_no('Do you wish to enable NVIDIA DLSS support (y/n)?\n'):
        return False
    print()
    license_bytes = urlopen('https://github.com/NVIDIA/DLSS/raw/refs/heads/main/LICENSE.txt').read()
    # Replace Unicode quotation marks with '"' and '\'' to avoid UnicodeDecodeError.
    license_bytes = license_bytes.replace(b'\x93', b'"').replace(b'\x94', b'"').replace(b'\x92', b'\'').replace(b'\x99', b'(TM)')
    license_string = license_bytes.decode('utf-8', errors='replace')
    print(f'{license_string}')
    if not prompt_yes_no('Accept NVIDIA DLSS license and download DLSS SDK (y/n)?\n'):
        return False
    print()
    subprocess.run([
        'git', 'clone', '--recurse-submodules',
        'https://github.com/NVIDIA/DLSS.git', 'third_party/DLSS'])
    print()
    return True


use_dlss = check_dlss()
if use_dlss:
    include_dirs.append('third_party/DLSS/include')
    source_files.append('src/Denoiser/DLSSDenoiser.cpp')
    defines.append(('SUPPORT_DLSS',))
    if IS_WINDOWS:
        extra_objects.append('third_party/DLSS/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib')
        if not uses_pip:
            data_files.append('third_party/DLSS/lib/Windows_x86_64/rel/nvngx_dlss.dll')
            data_files.append('third_party/DLSS/lib/Windows_x86_64/rel/nvngx_dlssd.dll')
    else:
        extra_objects.append('third_party/DLSS/lib/Linux_x86_64/libnvsdk_ngx.a')
        nvidia_ngx_so_path = glob.glob('third_party/DLSS/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.*')[0]
        nvidia_ngx_rr_so_path = glob.glob('third_party/DLSS/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.*')[0]
        if not uses_pip:
            data_files.append(nvidia_ngx_so_path)
            data_files.append(nvidia_ngx_rr_so_path)

data_files_all.append(('.', data_files))


def update_data_files_recursive(data_files_all, directory):
    files_in_directory = []
    for filename in os.listdir(directory):
        abs_file = directory + "/" + filename
        if os.path.isdir(abs_file):
            update_data_files_recursive(data_files_all, abs_file)
        else:
            files_in_directory.append(abs_file)
    if len(files_in_directory) > 0:
        data_files_all.append((directory, files_in_directory))


update_data_files_recursive(data_files_all, 'docs')
update_data_files_recursive(data_files_all, 'Data/Shaders')
update_data_files_recursive(data_files_all, 'Data/TransferFunctions')

for define in defines:
    if IS_WINDOWS:
        if len(define) == 1:
            extra_compile_args_cxx.append('/D')
            extra_compile_args_cxx.append(f'{define[0]}')
            extra_compile_args_nvcc.append(f'-D{define[0]}')
        else:
            extra_compile_args_cxx.append('/D')
            extra_compile_args_cxx.append(f'{define[0]}={define[1]}')
            extra_compile_args_nvcc.append(f'-D{define[0]}={define[1]}')
    else:
        if len(define) == 1:
            extra_compile_args.append(f'-D{define[0]}')
        else:
            extra_compile_args.append(f'-D{define[0]}={define[1]}')

extra_compile_args_cxx = extra_compile_args + extra_compile_args_cxx
extra_compile_args_nvcc = extra_compile_args + extra_compile_args_nvcc
extra_compile_args = {'cxx': extra_compile_args_cxx, 'nvcc': extra_compile_args_nvcc}

if uses_pip:
    if os.path.exists('vpt'):
        shutil.rmtree('vpt')
    Path('vpt/Data').mkdir(parents=True, exist_ok=True)
    shutil.copy('src/PyTorch/vpt.pyi', 'vpt/__init__.pyi')
    shutil.copy('LICENSE', 'vpt/LICENSE')
    shutil.copytree('docs', 'vpt/docs')
    shutil.copytree('Data/Shaders', 'vpt/Data/Shaders')
    shutil.copytree('Data/TransferFunctions', 'vpt/Data/TransferFunctions')
    pkg_data = ['**/LICENSE']
    if not IS_WINDOWS:
        patchelf_exec = shutil.which('patchelf')
    else:
        patchelf_exec = None
    for lib_path in oidn_shared_libs:
        lib_name = os.path.basename(lib_path)
        dest_lib_path = f'vpt/{lib_name}'
        shutil.copy(lib_path, dest_lib_path)
        if patchelf_exec is not None:
            subprocess.run([patchelf_exec, '--set-rpath', '\'$ORIGIN\'', dest_lib_path], check=True)
        pkg_data.append(f'**/{lib_name}')
    if use_dlss:
        if IS_WINDOWS:
            pkg_data.append('**/*.dll')
            shutil.copy('third_party/DLSS/lib/Windows_x86_64/rel/nvngx_dlss.dll', 'vpt/nvngx_dlss.dll')
            shutil.copy('third_party/DLSS/lib/Windows_x86_64/rel/nvngx_dlssd.dll', 'vpt/nvngx_dlssd.dll')
        else:
            pkg_data.append(f'**/{os.path.basename(nvidia_ngx_so_path)}')
            shutil.copy(nvidia_ngx_so_path, f'vpt/{os.path.basename(nvidia_ngx_so_path)}')
            pkg_data.append(f'**/{os.path.basename(nvidia_ngx_rr_so_path)}')
            shutil.copy(nvidia_ngx_rr_so_path, f'vpt/{os.path.basename(nvidia_ngx_rr_so_path)}')
    ext_modules = [
        CUDAExtension(
            'vpt.vpt',
            source_files,
            libraries=libraries,
            library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            extra_objects=extra_objects
        )
    ]
    dist = Distribution(attrs={'name': 'vpt', 'version': '0.0.0', 'ext_modules': ext_modules})
    bdist_egg_cmd = dist.get_command_obj('bdist_egg')
    build_cmd = bdist_egg_cmd.get_finalized_command('build_ext')
    vpt_so_file = ''
    for ext in build_cmd.extensions:
        fullname = build_cmd.get_ext_fullname(ext.name)
        filename = build_cmd.get_ext_filename(fullname)
        vpt_so_file = os.path.basename(filename)
    with open('vpt/__init__.py', 'w') as file:
        if not IS_WINDOWS:
            file.write('import ctypes\n')
        file.write('import torch\n\n')
        file.write('def __bootstrap__():\n')
        file.write('    global __bootstrap__, __loader__, __file__\n')
        file.write('    import sys, pkg_resources, importlib.util\n')
        file.write(f'    __file__ = pkg_resources.resource_filename(__name__, \'{vpt_so_file}\')\n')
        file.write('    __loader__ = None; del __bootstrap__, __loader__\n')
        if not IS_WINDOWS:
            file.write(f'    oidn_core = ctypes.cdll.LoadLibrary(pkg_resources.resource_filename(__name__, \'libOpenImageDenoise_core.so.{oidn_version}\'))\n')
            file.write('    oidn = ctypes.cdll.LoadLibrary(pkg_resources.resource_filename(__name__, \'libOpenImageDenoise.so.2\'))\n')
        file.write('    spec = importlib.util.spec_from_file_location(__name__,__file__)\n')
        file.write('    mod = importlib.util.module_from_spec(spec)\n')
        file.write('    spec.loader.exec_module(mod)\n')
        file.write('__bootstrap__()\n')
    setup(
        name='vpt',
        author='Christoph Neuhauser',
        ext_modules=ext_modules,
        packages=find_packages(include=['vpt', 'vpt.*']),
        package_data={'vpt': ['**/*.py', '**/*.pyi', '**/*.md', '**/*.txt', '**/*.xml', '**/*.glsl'] + pkg_data},
        #include_package_data=True,
        cmdclass={
            'build_ext': BuildExtension,
            'egg_info': EggInfoInstallLicense
        },
        license_files=('LICENSE',),
        include_dirs=include_dirs
    )
else:
    setup(
        name='vpt',
        author='Christoph Neuhauser',
        ext_modules=[
            CUDAExtension(
                'vpt',
                source_files,
                libraries=libraries,
                library_dirs=library_dirs,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                extra_objects=extra_objects
            )
        ],
        data_files=data_files_all,
        cmdclass={
            'build_ext': BuildExtension,
            'egg_info': EggInfoInstallLicense
        },
        license_files=('LICENSE',),
        include_dirs=include_dirs
    )

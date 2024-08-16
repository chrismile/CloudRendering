#!/usr/bin/env bash

# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Christoph Neuhauser
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

# Conda crashes with "set -euo pipefail".
set -eo pipefail

scriptpath="$( cd "$(dirname "$0")" ; pwd -P )"
projectpath="$scriptpath"
pushd $scriptpath > /dev/null

if [[ "$(uname -s)" =~ ^MSYS_NT.* ]] || [[ "$(uname -s)" =~ ^MINGW.* ]]; then
    use_msys=true
else
    use_msys=false
fi
if [[ "$(uname -s)" =~ ^Darwin.* ]]; then
    use_macos=true
else
    use_macos=false
fi
os_arch="$(uname -m)"

run_program=true
debug=false
glibcxx_debug=false
clean=false
build_dir_debug=".build_debug"
build_dir_release=".build_release"
use_vcpkg=false
use_conda=false
conda_env_name="cloudrendering"
link_dynamic=false
use_custom_vcpkg_triplet=false
custom_glslang=false
build_with_openvdb_support=false
use_pytorch=false
use_pre_cxx11_abi=false
install_module=false
use_download_swapchain=false
use_custom_jsoncpp=false
use_custom_openexr=false
use_open_image_denoise=true

# Check if a conda environment is already active.
if $use_conda; then
    if [ ! -z "${CONDA_DEFAULT_ENV+x}" ]; then
        conda_env_name="$CONDA_DEFAULT_ENV"
    fi
fi

# Process command line arguments.
for ((i=1;i<=$#;i++));
do
    if [ ${!i} = "--do-not-run" ]; then
        run_program=false
    elif [ ${!i} = "--debug" ] || [ ${!i} = "debug" ]; then
        debug=true
    elif [ ${!i} = "--glibcxx-debug" ]; then
        glibcxx_debug=true
    elif [ ${!i} = "--clean" ] || [ ${!i} = "clean" ]; then
        clean=true
    elif [ ${!i} = "--vcpkg" ] || [ ${!i} = "--use-vcpkg" ]; then
        use_vcpkg=true
    elif [ ${!i} = "--conda" ] || [ ${!i} = "--use-conda" ]; then
        use_conda=true
    elif [ ${!i} = "--conda-env-name" ]; then
        ((i++))
        conda_env_name=${!i}
    elif [ ${!i} = "--link-static" ]; then
        link_dynamic=false
    elif [ ${!i} = "--link-dynamic" ]; then
        link_dynamic=true
    elif [ ${!i} = "--vcpkg-triplet" ]; then
        ((i++))
        vcpkg_triplet=${!i}
        use_custom_vcpkg_triplet=true
    elif [ ${!i} = "--custom-glslang" ]; then
        custom_glslang=true
    elif [ ${!i} = "--use-openvdb" ]; then
      build_with_openvdb_support=true
    elif [ ${!i} = "--use-pytorch" ]; then
      use_pytorch=true
      pytorch_cmake_path="$(dirname $(python -c "import torch; print(torch.__file__)"))/share/cmake"
      pytorch_is_cxx11="$(python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)")"
      if [[ "$pytorch_is_cxx11" == "True" ]]; then
          use_pre_cxx11_abi=false
      elif [[ "$pytorch_is_cxx11" == "False" ]]; then
          # Theoretically pre-C++11 ABI support should work, but in practice, when linking using Boost installed via
          # Conda, this resulted in std::bad_alloc when the string constructor is called by boost::filesystem.
          #echo "Error: PyTorch is not built with CXX11 ABI." 1>&2
          #exit 1
          export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"
          use_pre_cxx11_abi=true
          use_custom_jsoncpp=true
          use_custom_openexr=true
          echo "Warning: PyTorch is not built with CXX11 ABI. Using compatibility mode."
      else
          echo "Error: PyTorch was not found." 1>&2
          exit 1
      fi
    elif [ ${!i} = "--install-dir" ]; then
        install_module=true
        ((i++))
        install_dir=${!i}
    elif [ ${!i} = "--dlswap" ]; then
        use_download_swapchain=true
        build_with_vkvg_support=true
    fi
done

if [ $clean = true ]; then
    echo "------------------------"
    echo " cleaning up old files  "
    echo "------------------------"
    rm -rf third_party/sgl/ third_party/vcpkg/ .build_release/ .build_debug/ Shipping/
    git submodule update --init --recursive
fi

if [ $debug = true ]; then
    cmake_config="Debug"
    build_dir=$build_dir_debug
else
    cmake_config="Release"
    build_dir=$build_dir_release
fi
destination_dir="Shipping"
if $use_macos; then
    binaries_dest_dir="$destination_dir/CloudRendering.app/Contents/MacOS"
    if ! command -v brew &> /dev/null; then
        if [ ! -d "/opt/homebrew/bin" ]; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        if [ -d "/opt/homebrew/bin" ]; then
            #echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/$USER/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    fi
fi

params_link=()
params_vcpkg=()
build_sgl_release_only=false
if [ $use_custom_vcpkg_triplet = true ]; then
    params_link+=(-DVCPKG_TARGET_TRIPLET=$vcpkg_triplet)
    if [ -f "$projectpath/third_party/vcpkg/triplets/$vcpkg_triplet.cmake" ]; then
        triplet_file_path="$projectpath/third_party/vcpkg/triplets/$vcpkg_triplet.cmake"
    elif [ -f "$projectpath/third_party/vcpkg/triplets/community/$vcpkg_triplet.cmake" ]; then
        triplet_file_path="$projectpath/third_party/vcpkg/triplets/community/$vcpkg_triplet.cmake"
    else
        echo "Custom vcpkg triplet set, but file not found."
        exit 1
    fi
    if grep -q "VCPKG_BUILD_TYPE release" "$triplet_file_path"; then
        build_sgl_release_only=true
    fi
elif [ $use_vcpkg = true ] && [ $use_macos = false ] && [ $link_dynamic = true ]; then
    params_link+=(-DVCPKG_TARGET_TRIPLET=x64-linux-dynamic)
fi
if [ $use_vcpkg = true ] && [ $use_macos = false ]; then
    params_vcpkg+=(-DUSE_STATIC_STD_LIBRARIES=On)
fi
if [ $use_vcpkg = true ]; then
    params_vcpkg+=(-DCMAKE_TOOLCHAIN_FILE="$projectpath/third_party/vcpkg/scripts/buildsystems/vcpkg.cmake")
fi

is_installed_apt() {
    local pkg_name="$1"
    if [ "$(dpkg -l | awk '/'"$pkg_name"'/ {print }'|wc -l)" -ge 1 ]; then
        return 0
    else
        return 1
    fi
}

is_installed_pacman() {
    local pkg_name="$1"
    if pacman -Qs $pkg_name > /dev/null; then
        return 0
    else
        return 1
    fi
}

is_installed_yay() {
    local pkg_name="$1"
    if yay -Ss $pkg_name > /dev/null | grep -q 'instal'; then
        return 1
    else
        return 0
    fi
}

is_installed_yum() {
    local pkg_name="$1"
    if yum list installed "$pkg_name" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

is_installed_rpm() {
    local pkg_name="$1"
    if rpm -q "$pkg_name" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

is_installed_brew() {
    local pkg_name="$1"
    if brew list $pkg_name > /dev/null; then
        return 0
    else
        return 1
    fi
}

# https://stackoverflow.com/questions/8063228/check-if-a-variable-exists-in-a-list-in-bash
list_contains() {
    if [[ "$1" =~ (^|[[:space:]])"$2"($|[[:space:]]) ]]; then
        return 0
    else
        return 1
    fi
}

if $use_msys && command -v pacman &> /dev/null && [ ! -d $build_dir_debug ] && [ ! -d $build_dir_release ]; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v rsync &> /dev/null \
            || ! command -v curl &> /dev/null || ! command -v wget &> /dev/null || ! command -v unzip &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null \
            || ! command -v ntldd &> /dev/null; then
        echo "------------------------"
        echo "installing build essentials"
        echo "------------------------"
        pacman --noconfirm -S --needed make git rsync curl wget unzip mingw64/mingw-w64-x86_64-cmake \
        mingw64/mingw-w64-x86_64-gcc mingw64/mingw-w64-x86_64-gdb mingw-w64-x86_64-ntldd
    fi

    # Dependencies of sgl and the application.
    if ! is_installed_pacman "mingw-w64-x86_64-boost" || ! is_installed_pacman "mingw-w64-x86_64-icu" \
            || ! is_installed_pacman "mingw-w64-x86_64-glm" || ! is_installed_pacman "mingw-w64-x86_64-libarchive" \
            || ! is_installed_pacman "mingw-w64-x86_64-tinyxml2" || ! is_installed_pacman "mingw-w64-x86_64-libpng" \
            || ! is_installed_pacman "mingw-w64-x86_64-SDL2" || ! is_installed_pacman "mingw-w64-x86_64-SDL2_image" \
            || ! is_installed_pacman "mingw-w64-x86_64-glew" || ! is_installed_pacman "mingw-w64-x86_64-vulkan-headers" \
            || ! is_installed_pacman "mingw-w64-x86_64-vulkan-loader" \
            || ! is_installed_pacman "mingw-w64-x86_64-vulkan-validation-layers" \
            || ! is_installed_pacman "mingw-w64-x86_64-shaderc" \
            || ! is_installed_pacman "mingw-w64-x86_64-opencl-headers" \
            || ! is_installed_pacman "mingw-w64-x86_64-opencl-icd" || ! is_installed_pacman "mingw-w64-x86_64-jsoncpp" \
            || ! is_installed_pacman "mingw-w64-x86_64-openexr" || ! is_installed_pacman "mingw-w64-x86_64-openvdb" \
            || ! is_installed_pacman "mingw-w64-x86_64-tbb" || ! is_installed_pacman "mingw-w64-x86_64-blosc"; then
        echo "------------------------"
        echo "installing dependencies "
        echo "------------------------"
        pacman --noconfirm -S --needed mingw64/mingw-w64-x86_64-boost mingw64/mingw-w64-x86_64-icu \
        mingw64/mingw-w64-x86_64-glm mingw64/mingw-w64-x86_64-libarchive mingw64/mingw-w64-x86_64-tinyxml2 \
        mingw64/mingw-w64-x86_64-libpng mingw64/mingw-w64-x86_64-SDL2 mingw64/mingw-w64-x86_64-SDL2_image \
        mingw64/mingw-w64-x86_64-glew mingw64/mingw-w64-x86_64-vulkan-headers mingw64/mingw-w64-x86_64-vulkan-loader \
        mingw64/mingw-w64-x86_64-vulkan-validation-layers mingw64/mingw-w64-x86_64-shaderc \
        mingw64/mingw-w64-x86_64-opencl-headers mingw64/mingw-w64-x86_64-opencl-icd mingw64/mingw-w64-x86_64-jsoncpp \
        mingw64/mingw-w64-x86_64-openexr mingw64/mingw-w64-x86_64-openvdb mingw64/mingw-w64-x86_64-tbb \
        mingw64/mingw-w64-x86_64-blosc
    fi
elif $use_msys && command -v pacman &> /dev/null; then
    :
elif [ ! -z "${BUILD_USE_NIX+x}" ]; then
    echo "------------------------"
    echo "   building using Nix"
    echo "------------------------"
elif $use_macos && command -v brew &> /dev/null && [ ! -d $build_dir_debug ] && [ ! -d $build_dir_release ]; then
    if ! is_installed_brew "git"; then
        brew install git
    fi
    if ! is_installed_brew "cmake"; then
        brew install cmake
    fi
    if ! is_installed_brew "curl"; then
        brew install curl
    fi
    if ! is_installed_brew "pkg-config"; then
        brew install pkg-config
    fi
    if ! is_installed_brew "llvm"; then
        brew install llvm
    fi
    if ! is_installed_brew "libomp"; then
        brew install libomp
    fi
    if ! is_installed_brew "autoconf"; then
        brew install autoconf
    fi
    if ! is_installed_brew "automake"; then
        brew install automake
    fi
    if ! is_installed_brew "autoconf-archive"; then
        brew install autoconf-archive
    fi

    # Homebrew MoltenVK does not contain script for setting environment variables, unfortunately.
    #if ! is_installed_brew "molten-vk"; then
    #    brew install molten-vk
    #fi

    # Dependencies of sgl and the application.
    if [ $use_vcpkg = false ]; then
        if ! is_installed_brew "boost"; then
            brew install boost
        fi
        if ! is_installed_brew "icu4c"; then
            brew install icu4c
        fi
        if ! is_installed_brew "glm"; then
            brew install glm
        fi
        if ! is_installed_brew "libarchive"; then
            brew install libarchive
        fi
        if ! is_installed_brew "tinyxml2"; then
            brew install tinyxml2
        fi
        if ! is_installed_brew "zlib"; then
            brew install zlib
        fi
        if ! is_installed_brew "libpng"; then
            brew install libpng
        fi
        if ! is_installed_brew "sdl2"; then
            brew install sdl2
        fi
        if ! is_installed_brew "sdl2_image"; then
            brew install sdl2_image
        fi
        if ! is_installed_brew "glew"; then
            brew install glew
        fi
        if ! is_installed_brew "opencl-headers"; then
            brew install opencl-headers
        fi
        if ! is_installed_brew "jsoncpp"; then
            brew install jsoncpp
        fi
        if ! is_installed_brew "openexr"; then
            brew install openexr
        fi
        if ! is_installed_brew "tbb"; then
            brew install tbb
        fi
        if ! is_installed_brew "c-blosc"; then
            brew install c-blosc
        fi
    fi
elif $use_macos && command -v brew &> /dev/null; then
    :
elif command -v apt &> /dev/null && ! $use_conda; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v curl &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null \
            || ! command -v patchelf &> /dev/null; then
        echo "------------------------"
        echo "installing build essentials"
        echo "------------------------"
        sudo apt install -y cmake git curl pkg-config build-essential patchelf
    fi

    # Dependencies of sgl and the application.
    if $use_vcpkg; then
        if ! is_installed_apt "libgl-dev" || ! is_installed_apt "libxmu-dev" || ! is_installed_apt "libxi-dev" \
                || ! is_installed_apt "libx11-dev" || ! is_installed_apt "libxft-dev" \
                || ! is_installed_apt "libxext-dev" || ! is_installed_apt "libxrandr-dev" \
                || ! is_installed_apt "libwayland-dev" || ! is_installed_apt "libxkbcommon-dev" \
                || ! is_installed_apt "libegl1-mesa-dev" || ! is_installed_apt "libibus-1.0-dev" \
                || ! is_installed_apt "autoconf" || ! is_installed_apt "automake" \
                || ! is_installed_apt "autoconf-archive"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo apt install -y libgl-dev libxmu-dev libxi-dev libx11-dev libxft-dev libxext-dev libxrandr-dev \
            libwayland-dev libxkbcommon-dev libegl1-mesa-dev libibus-1.0-dev autoconf automake autoconf-archive
        fi
    else
        if ! is_installed_apt "libboost-filesystem-dev" || ! is_installed_apt "libicu-dev" \
                || ! is_installed_apt "libglm-dev" || ! is_installed_apt "libarchive-dev" \
                || ! is_installed_apt "libtinyxml2-dev" || ! is_installed_apt "libpng-dev" \
                || ! is_installed_apt "libsdl2-dev" || ! is_installed_apt "libsdl2-image-dev" \
                || ! is_installed_apt "libglew-dev" || ! is_installed_apt "opencl-c-headers" \
                || ! is_installed_apt "ocl-icd-opencl-dev" || ! is_installed_apt "libjsoncpp-dev" \
                || ! is_installed_apt "libopenexr-dev" || ! is_installed_apt "libboost-iostreams-dev" \
                || ! is_installed_apt "libtbb-dev" || ! is_installed_apt "libblosc-dev" \
                || ! is_installed_apt "liblz4-dev"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo apt install -y libboost-filesystem-dev libicu-dev libglm-dev libarchive-dev libtinyxml2-dev libpng-dev \
            libsdl2-dev libsdl2-image-dev libglew-dev opencl-c-headers ocl-icd-opencl-dev libjsoncpp-dev libopenexr-dev \
            libboost-iostreams-dev libtbb-dev libblosc-dev liblz4-dev
        fi
    fi
elif command -v pacman &> /dev/null && ! $use_conda; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v curl &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null \
            || ! command -v patchelf &> /dev/null; then
        echo "------------------------"
        echo "installing build essentials"
        echo "------------------------"
        sudo pacman -S cmake git curl pkgconf base-devel patchelf
    fi

    # Dependencies of sgl and the application.
    if $use_vcpkg; then
        if ! is_installed_pacman "libgl" || ! is_installed_pacman "vulkan-devel" || ! is_installed_pacman "shaderc" \
                || ! is_installed_pacman "openssl" || ! is_installed_pacman "autoconf" \
                || ! is_installed_pacman "automake" || ! is_installed_pacman "autoconf-archive"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo pacman -S libgl vulkan-devel shaderc openssl autoconf automake autoconf-archive
        fi
    else
        if ! is_installed_pacman "boost" || ! is_installed_pacman "icu" || ! is_installed_pacman "glm" \
                || ! is_installed_pacman "libarchive" || ! is_installed_pacman "tinyxml2" \
                || ! is_installed_pacman "libpng" || ! is_installed_pacman "sdl2" || ! is_installed_pacman "sdl2_image" \
                || ! is_installed_pacman "glew" || ! is_installed_pacman "vulkan-devel" \
                || ! is_installed_pacman "shaderc" || ! is_installed_pacman "opencl-headers" \
                || ! is_installed_pacman "ocl-icd" || ! is_installed_pacman "jsoncpp" || ! is_installed_pacman "openexr" \
                || ! is_installed_pacman "onetbb" || ! is_installed_pacman "blosc"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo pacman -S boost icu glm libarchive tinyxml2 libpng sdl2 sdl2_image glew vulkan-devel shaderc \
            opencl-headers ocl-icd jsoncpp openexr onetbb blosc
        fi
    fi
elif command -v yum &> /dev/null && ! $use_conda; then
    if ! command -v cmake &> /dev/null || ! command -v git &> /dev/null || ! command -v curl &> /dev/null \
            || ! command -v pkg-config &> /dev/null || ! command -v g++ &> /dev/null \
            || ! command -v patchelf &> /dev/null; then
        echo "------------------------"
        echo "installing build essentials"
        echo "------------------------"
        sudo yum install -y cmake git curl pkgconf gcc gcc-c++ patchelf
    fi

    # Dependencies of sgl and the application.
    if $use_vcpkg; then
        if ! is_installed_rpm "perl" || ! is_installed_rpm "libstdc++-devel" || ! is_installed_rpm "libstdc++-static" \
                || ! is_installed_rpm "autoconf" || ! is_installed_rpm "automake" \
                || ! is_installed_rpm "autoconf-archive" || ! is_installed_rpm "glew-devel" \
                || ! is_installed_rpm "libXext-devel" || ! is_installed_rpm "vulkan-headers" \
                || ! is_installed_rpm "vulkan-loader" || ! is_installed_rpm "vulkan-tools" \
                || ! is_installed_rpm "vulkan-validation-layers" || ! is_installed_rpm "libshaderc-devel"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo yum install -y perl libstdc++-devel libstdc++-static autoconf automake autoconf-archive glew-devel \
            libXext-devel vulkan-headers vulkan-loader vulkan-tools vulkan-validation-layers libshaderc-devel
        fi
    else
        if ! is_installed_rpm "boost-devel" || ! is_installed_rpm "libicu-devel" || ! is_installed_rpm "glm-devel" \
                || ! is_installed_rpm "libarchive-devel" || ! is_installed_rpm "tinyxml2-devel" \
                || ! is_installed_rpm "libpng-devel" || ! is_installed_rpm "SDL2-devel" \
                || ! is_installed_rpm "SDL2_image-devel" || ! is_installed_rpm "glew-devel" \
                || ! is_installed_rpm "vulkan-headers" || ! is_installed_rpm "libshaderc-devel" \
                || ! is_installed_rpm "opencl-headers" || ! is_installed_rpm "ocl-icd" \
                || ! is_installed_rpm "jsoncpp-devel" || ! is_installed_rpm "openexr-devel" \
                || ! is_installed_rpm "tbb-devel" || ! is_installed_rpm "blosc-devel"; then
            echo "------------------------"
            echo "installing dependencies "
            echo "------------------------"
            sudo yum install -y boost-devel libicu-devel glm-devel libarchive-devel tinyxml2-devel libpng-devel \
            SDL2-devel SDL2_image-devel glew-devel vulkan-headers libshaderc-devel opencl-headers ocl-icd jsoncpp-devel \
            openexr-devel tbb-devel blosc-devel
        fi
    fi
elif $use_conda && ! $use_macos; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh" shell.bash hook
    elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh" shell.bash hook
    elif [ ! -z "${CONDA_PREFIX+x}" ]; then
        . "$CONDA_PREFIX/etc/profile.d/conda.sh" shell.bash hook
    fi

    if ! command -v conda &> /dev/null; then
        echo "------------------------"
        echo "  installing Miniconda  "
        echo "------------------------"
        if [ "$os_arch" = "x86_64" ]; then
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
            chmod +x Miniconda3-latest-Linux-x86_64.sh
            bash ./Miniconda3-latest-Linux-x86_64.sh
            . "$HOME/miniconda3/etc/profile.d/conda.sh" shell.bash hook
            rm ./Miniconda3-latest-Linux-x86_64.sh
        else
            wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
            chmod +x Miniconda3-latest-Linux-aarch64.sh
            bash ./Miniconda3-latest-Linux-aarch64.sh
            . "$HOME/miniconda3/etc/profile.d/conda.sh" shell.bash hook
            rm ./Miniconda3-latest-Linux-aarch64.sh
        fi
    fi

    if ! conda env list | grep ".*${conda_env_name}.*" >/dev/null 2>&1; then
        echo "------------------------"
        echo "creating conda environment"
        echo "------------------------"
        conda create -n "${conda_env_name}" -y
        conda init bash
        conda activate "${conda_env_name}"
    elif [ "${var+CONDA_DEFAULT_ENV}" != "${conda_env_name}" ]; then
        conda activate "${conda_env_name}"
    fi

    conda_pkg_list="$(conda list)"
    if ! list_contains "$conda_pkg_list" "boost" || ! list_contains "$conda_pkg_list" "conda-forge::icu" \
            || ! list_contains "$conda_pkg_list" "glm" || ! list_contains "$conda_pkg_list" "libarchive" \
            || ! list_contains "$conda_pkg_list" "tinyxml2" || ! list_contains "$conda_pkg_list" "libpng" \
            || ! list_contains "$conda_pkg_list" "sdl2" || ! list_contains "$conda_pkg_list" "sdl2" \
            || ! list_contains "$conda_pkg_list" "glew" || ! list_contains "$conda_pkg_list" "cxx-compiler" \
            || ! list_contains "$conda_pkg_list" "make" || ! list_contains "$conda_pkg_list" "cmake" \
            || ! list_contains "$conda_pkg_list" "pkg-config" || ! list_contains "$conda_pkg_list" "gdb" \
            || ! list_contains "$conda_pkg_list" "git" \
            || ! list_contains "$conda_pkg_list" "mesa-libgl-devel-cos7-x86_64" \
            || ! list_contains "$conda_pkg_list" "libglvnd-glx-cos7-x86_64" \
            || ! list_contains "$conda_pkg_list" "mesa-dri-drivers-cos7-aarch64" \
            || ! list_contains "$conda_pkg_list" "libxau-devel-cos7-aarch64" \
            || ! list_contains "$conda_pkg_list" "libselinux-devel-cos7-aarch64" \
            || ! list_contains "$conda_pkg_list" "libxdamage-devel-cos7-aarch64" \
            || ! list_contains "$conda_pkg_list" "libxxf86vm-devel-cos7-aarch64" \
            || ! list_contains "$conda_pkg_list" "libxext-devel-cos7-aarch64" \
            || ! list_contains "$conda_pkg_list" "xorg-libxfixes" || ! list_contains "$conda_pkg_list" "xorg-libxau" \
            || ! list_contains "$conda_pkg_list" "xorg-libxrandr" || ! list_contains "$conda_pkg_list" "patchelf" \
            || ! list_contains "$conda_pkg_list" "libvulkan-headers" || ! list_contains "$conda_pkg_list" "shaderc" \
            || ! list_contains "$conda_pkg_list" "jsoncpp" || ! list_contains "$conda_pkg_list" "openexr" \
            || ! list_contains "$conda_pkg_list" "conda-forge::tbb" \
            || ! list_contains "$conda_pkg_list" "conda-forge::tbb-devel" \
            || ! list_contains "$conda_pkg_list" "blosc"; then
        echo "------------------------"
        echo "installing dependencies "
        echo "------------------------"
        conda install -y -c conda-forge boost conda-forge::icu glm libarchive tinyxml2 libpng sdl2 sdl2 glew \
        cxx-compiler make cmake pkg-config gdb git mesa-libgl-devel-cos7-x86_64 libglvnd-glx-cos7-x86_64 \
        mesa-dri-drivers-cos7-aarch64 libxau-devel-cos7-aarch64 libselinux-devel-cos7-aarch64 \
        libxdamage-devel-cos7-aarch64 libxxf86vm-devel-cos7-aarch64 libxext-devel-cos7-aarch64 xorg-libxfixes \
        xorg-libxau xorg-libxrandr patchelf libvulkan-headers shaderc jsoncpp openexr conda-forge::tbb \
        conda-forge::tbb-devel blosc
    fi
else
    echo "Warning: Unsupported system package manager detected." >&2
fi

if ! command -v cmake &> /dev/null; then
    echo "CMake was not found, but is required to build the program."
    exit 1
fi
if ! command -v git &> /dev/null; then
    echo "git was not found, but is required to build the program."
    exit 1
fi
if ! command -v curl &> /dev/null; then
    echo "curl was not found, but is required to build the program."
    exit 1
fi
if [ $use_macos = false ] && ! command -v pkg-config &> /dev/null; then
    echo "pkg-config was not found, but is required to build the program."
    exit 1
fi

if [ ! -d "submodules/IsosurfaceCpp/src" ]; then
    echo "------------------------"
    echo "initializing submodules "
    echo "------------------------"
    git submodule init
    git submodule update
fi

[ -d "./third_party/" ] || mkdir "./third_party/"
pushd third_party > /dev/null


params_sgl=()
params=()
params_run=()
params_gen=()

if [ $use_msys = true ]; then
    params_gen+=(-G "MSYS Makefiles")
    params_sgl+=(-G "MSYS Makefiles")
    params+=(-G "MSYS Makefiles")
fi

if [ $use_vcpkg = false ] && [ $use_macos = true ]; then
    params_gen+=(-DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=False)
    params_gen+=(-DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=False)
    params_gen+=(-DCMAKE_FIND_FRAMEWORK=LAST)
    params_gen+=(-DCMAKE_FIND_APPBUNDLE=NEVER)
    params_gen+=(-DCMAKE_PREFIX_PATH="$(brew --prefix)")
    params_sgl+=(-DCMAKE_INSTALL_PREFIX="../install")
    params_sgl+=(-DZLIB_ROOT="$(brew --prefix)/opt/zlib")
    params+=(-DZLIB_ROOT="$(brew --prefix)/opt/zlib")
fi

if $glibcxx_debug; then
    params_sgl+=(-DUSE_GLIBCXX_DEBUG=On)
    params+=(-DUSE_GLIBCXX_DEBUG=On)
fi

if [ $use_pytorch = true ]; then
    params+=(-DCMAKE_PREFIX_PATH="$pytorch_cmake_path")
    params+=(-DBUILD_PYTORCH_MODULE=On -DSUPPORT_PYTORCH_DENOISER=On -DBUILD_KPN_MODULE=On)
    if [ $install_module = true ]; then
        params+=(-DCMAKE_INSTALL_PREFIX="$install_dir")
    fi
    if [ $use_pre_cxx11_abi = true ]; then
        # Theoretically pre-C++11 ABI support should work, but in practice, when linking using Boost installed via
        # Conda, this resulted in std::bad_alloc when the string constructor is called by boost::filesystem.
        params_sgl+=(-DUSE_PRE_CXX11_ABI=ON -DUSE_BOOST=OFF)
        params+=(-DUSE_PRE_CXX11_ABI=ON)
    fi
fi

use_vulkan=false
vulkan_sdk_env_set=true
use_vulkan=true

search_for_vulkan_sdk=false
if [ $use_msys = false ] && [ -z "${VULKAN_SDK+1}" ]; then
    search_for_vulkan_sdk=true
fi

if [ $search_for_vulkan_sdk = true ]; then
    echo "------------------------"
    echo "searching for Vulkan SDK"
    echo "------------------------"

    found_vulkan=false
    use_local_vulkan_sdk=false

    if [ $use_macos = false ]; then
        if [ -d "VulkanSDK" ]; then
            VK_LAYER_PATH=""
            source "VulkanSDK/$(ls VulkanSDK)/setup-env.sh"
            use_local_vulkan_sdk=true
            pkgconfig_dir="$(realpath "VulkanSDK/$(ls VulkanSDK)/$os_arch/lib/pkgconfig")"
            if [ -d "$pkgconfig_dir" ]; then
                export PKG_CONFIG_PATH="$pkgconfig_dir"
            fi
            found_vulkan=true
        fi

        if ! $found_vulkan && (lsb_release -a 2> /dev/null | grep -q 'Ubuntu' || lsb_release -a 2> /dev/null | grep -q 'Mint') && ! $use_conda; then
            if lsb_release -a 2> /dev/null | grep -q 'Ubuntu'; then
                distro_code_name=$(lsb_release -cs)
                distro_release=$(lsb_release -rs)
            else
                distro_code_name=$(cat /etc/upstream-release/lsb-release | grep "DISTRIB_CODENAME=" | sed 's/^.*=//')
                distro_release=$(cat /etc/upstream-release/lsb-release | grep "DISTRIB_RELEASE=" | sed 's/^.*=//')
            fi
            if ! compgen -G "/etc/apt/sources.list.d/lunarg-vulkan-*" > /dev/null \
                  && ! curl -s -I "https://packages.lunarg.com/vulkan/dists/${distro_code_name}/" | grep "2 404" > /dev/null; then
                echo "Setting up Vulkan SDK for $(lsb_release -ds)..."
                wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
                sudo curl --silent --show-error --fail \
                https://packages.lunarg.com/vulkan/lunarg-vulkan-${distro_code_name}.list \
                --output /etc/apt/sources.list.d/lunarg-vulkan-${distro_code_name}.list
                sudo apt update
                sudo apt install -y vulkan-sdk shaderc glslang-dev
            elif dpkg --compare-versions "$distro_release" "ge" "24.04"; then
                if ! is_installed_apt "libvulkan-dev" || ! is_installed_apt "libshaderc-dev" || ! is_installed_apt "glslang-dev"; then
                    # Optional: vulkan-validationlayers
                    sudo apt install -y libvulkan-dev libshaderc-dev glslang-dev
                fi
            fi
        fi

        if [ -d "/usr/include/vulkan" ] && [ -d "/usr/include/shaderc" ]; then
            if ! grep -q VULKAN_SDK ~/.bashrc; then
                echo 'export VULKAN_SDK="/usr"' >> ~/.bashrc
            fi
            export VULKAN_SDK="/usr"
            found_vulkan=true
        fi

        if ! $found_vulkan; then
            curl --silent --show-error --fail -O https://sdk.lunarg.com/sdk/download/latest/linux/vulkan-sdk.tar.gz
            mkdir -p VulkanSDK
            tar -xf vulkan-sdk.tar.gz -C VulkanSDK
            if [ "$os_arch" != "x86_64" ]; then
                pushd "VulkanSDK/$(ls VulkanSDK)" >/dev/null
                ./vulkansdk -j $(nproc) vulkan-loader glslang shaderc
                popd >/dev/null
            fi
            VK_LAYER_PATH=""
            source "VulkanSDK/$(ls VulkanSDK)/setup-env.sh"
            use_local_vulkan_sdk=true

            # Fix pkgconfig file.
            shaderc_pkgconfig_file="VulkanSDK/$(ls VulkanSDK)/$os_arch/lib/pkgconfig/shaderc.pc"
            if [ -f "$shaderc_pkgconfig_file" ]; then
                prefix_path=$(realpath "VulkanSDK/$(ls VulkanSDK)/$os_arch")
                sed -i '3s;.*;prefix=\"'$prefix_path'\";' "$shaderc_pkgconfig_file"
                sed -i '5s;.*;libdir=${prefix}/lib;' "$shaderc_pkgconfig_file"
                export PKG_CONFIG_PATH="$(realpath "VulkanSDK/$(ls VulkanSDK)/$os_arch/lib/pkgconfig")"
            fi
            found_vulkan=true
        fi
    else
        if [ -d "$HOME/VulkanSDK" ] && [ ! -z "$(ls -A "$HOME/VulkanSDK")" ]; then
            source "$HOME/VulkanSDK/$(ls $HOME/VulkanSDK)/setup-env.sh"
            found_vulkan=true
        else
            vulkansdk_filename=$(curl -sIkL https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.dmg | sed -r '/filename=/!d;s/.*filename=(.*)$/\1/')
            VULKAN_SDK_VERSION=$(echo $vulkansdk_filename | sed -r 's/^.*vulkansdk-macos-(.*)\.dmg.*$/\1/')
            curl -O https://sdk.lunarg.com/sdk/download/latest/mac/vulkan-sdk.dmg
            sudo hdiutil attach vulkan-sdk.dmg
            # The directory was changed from '/Volumes/VulkanSDK' to, e.g., 'vulkansdk-macos-1.3.261.0'.
            vulkan_dir=$(find /Volumes -maxdepth 1 -name '[Vv]ulkan*' -not -path "/Volumes/VMware*" || true)
            sudo "${vulkan_dir}/InstallVulkan.app/Contents/MacOS/InstallVulkan" \
            --root ~/VulkanSDK/$VULKAN_SDK_VERSION --accept-licenses --default-answer --confirm-command install
            pushd ~/VulkanSDK/$VULKAN_SDK_VERSION
            sudo python3 ./install_vulkan.py || true
            popd
            sudo hdiutil unmount "${vulkan_dir}"
            source "$HOME/VulkanSDK/$(ls $HOME/VulkanSDK)/setup-env.sh"
            found_vulkan=true
        fi
    fi

    if ! $found_vulkan; then
        if [ $use_macos = false ]; then
            os_name="linux"
        else
            os_name="mac"
        fi
        echo "The environment variable VULKAN_SDK is not set but is required in the installation process."
        echo "Please refer to https://vulkan.lunarg.com/sdk/home#${os_name} for instructions on how to install the Vulkan SDK."
        exit 1
    fi

    # On RHEL 8.6, I got the following errors when using the libvulkan.so provided by the SDK:
    # dlopen failed: /lib64/libm.so.6: version `GLIBC_2.29' not found (required by /home/u12458/Correrender/third_party/VulkanSDK/1.3.275.0/x86_64/lib/libvulkan.so)
    # Thus, we should remove it from the path if necessary.
    if $use_local_vulkan_sdk; then
        pushd VulkanSDK/$(ls VulkanSDK)/$os_arch/lib >/dev/null
        if ldd -r libvulkan.so | grep "undefined symbol"; then
            echo "Removing Vulkan SDK libvulkan.so from path..."
            export LD_LIBRARY_PATH=$(echo ${LD_LIBRARY_PATH} | awk -v RS=: -v ORS=: '/VulkanSDK/ {next} {print}' | sed 's/:*$//') && echo $OUTPATH
        fi
        popd >/dev/null
    fi
fi

if $custom_glslang; then
    if [ ! -d "./glslang" ]; then
        echo "------------------------"
        echo "  downloading glslang   "
        echo "------------------------"
        # Make sure we have no leftovers from a failed build attempt.
        if [ -d "./glslang-src" ]; then
            rm -rf "./glslang-src"
        fi
        git clone https://github.com/KhronosGroup/glslang.git glslang-src
        pushd glslang-src >/dev/null
        ./update_glslang_sources.py
        mkdir build
        pushd build >/dev/null
        cmake ${params_gen[@]+"${params_gen[@]}"} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/glslang" ..
        make -j $(nproc)
        make install
        popd >/dev/null
        popd >/dev/null
    fi
    params_sgl+=(-Dglslang_DIR="${projectpath}/third_party/glslang" -DUSE_SHADERC=Off)
fi

if [ $use_msys = false ] && [ -z "${VULKAN_SDK+1}" ]; then
    vulkan_sdk_env_set=true
fi

if [ $use_vcpkg = true ] && [ ! -d "./vcpkg" ]; then
    echo "------------------------"
    echo "    fetching vcpkg      "
    echo "------------------------"
    if $use_vulkan && [ $vulkan_sdk_env_set = false ]; then
        echo "The environment variable VULKAN_SDK is not set but is required in the installation process."
        exit 1
    fi
    git clone --depth 1 https://github.com/microsoft/vcpkg.git
    vcpkg/bootstrap-vcpkg.sh -disableMetrics
    vcpkg/vcpkg install
fi

if [ $use_vcpkg = true ] && [ $use_macos = false ] && [ $link_dynamic = false ]; then
    params_sgl+=(-DBUILD_STATIC_LIBRARY=On)
fi

if [ ! -d "./sgl" ]; then
    echo "------------------------"
    echo "     fetching sgl       "
    echo "------------------------"
    git clone --depth 1 https://github.com/chrismile/sgl.git
fi

if [ -f "./sgl/$build_dir/CMakeCache.txt" ]; then
    remove_build_cache_sgl=false
    if grep -q vcpkg_installed "./sgl/$build_dir/CMakeCache.txt"; then
        cache_uses_vcpkg=true
    else
        cache_uses_vcpkg=false
    fi
    if ([ $use_vcpkg = true ] && [ $cache_uses_vcpkg = false ]) || ([ $use_vcpkg = false ] && [ $cache_uses_vcpkg = true ]); then
        remove_build_cache_sgl=true
    fi
    if grep -q "USE_PRE_CXX11_ABI:BOOL=ON" "./sgl/$build_dir/CMakeCache.txt"; then
        cache_uses_pre_cxx11_abi=true
    else
        cache_uses_pre_cxx11_abi=false
    fi
    if ([ $use_pre_cxx11_abi = true ] && [ $cache_uses_pre_cxx11_abi = false ]) || ([ $use_pre_cxx11_abi = false ] && [ $cache_uses_pre_cxx11_abi = true ]); then
        remove_build_cache_sgl=true
    fi
    if [ $remove_build_cache_sgl = true ]; then
        echo "Removing old sgl build cache..."
        if [ -d "./sgl/$build_dir_debug" ]; then
            rm -rf "./sgl/$build_dir_debug"
        fi
        if [ -d "./sgl/$build_dir_release" ]; then
            rm -rf "./sgl/$build_dir_release"
        fi
        if [ -d "./sgl/install" ]; then
            rm -rf "./sgl/install"
        fi
    fi
fi

if [ ! -d "./sgl/install" ]; then
    echo "------------------------"
    echo "     building sgl       "
    echo "------------------------"

    pushd "./sgl" >/dev/null
    if ! $build_sgl_release_only; then
        mkdir -p $build_dir_debug
    fi
    mkdir -p $build_dir_release

    if ! $build_sgl_release_only; then
        pushd "$build_dir_debug" >/dev/null
        cmake .. \
             -DCMAKE_BUILD_TYPE=Debug \
             -DCMAKE_INSTALL_PREFIX="../install" \
             ${params_gen[@]+"${params_gen[@]}"} ${params_link[@]+"${params_link[@]}"} \
             ${params_vcpkg[@]+"${params_vcpkg[@]}"} ${params_sgl[@]+"${params_sgl[@]}"}
        if [ $use_vcpkg = false ] && [ $use_macos = false ]; then
            make -j $(nproc)
            make install
        fi
        popd >/dev/null
    fi

    pushd $build_dir_release >/dev/null
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="../install" \
         ${params_gen[@]+"${params_gen[@]}"} ${params_link[@]+"${params_link[@]}"} \
         ${params_vcpkg[@]+"${params_vcpkg[@]}"} ${params_sgl[@]+"${params_sgl[@]}"}
    if [ $use_vcpkg = false ] && [ $use_macos = false ]; then
        make -j $(nproc)
        make install
    fi
    popd >/dev/null

    if [ $use_macos = true ]; then
        if ! $build_sgl_release_only; then
            cmake --build $build_dir_debug --parallel $(sysctl -n hw.ncpu)
            cmake --build $build_dir_debug --target install
        fi

        cmake --build $build_dir_release --parallel $(sysctl -n hw.ncpu)
        cmake --build $build_dir_release --target install
    elif [ $use_vcpkg = true ]; then
        if ! $build_sgl_release_only; then
            cmake --build $build_dir_debug --parallel $(nproc)
            cmake --build $build_dir_debug --target install
            if [ $link_dynamic = true ]; then
                cp $build_dir_debug/libsgld.so install/lib/libsgld.so
            fi
        fi

        cmake --build $build_dir_release --parallel $(nproc)
        cmake --build $build_dir_release --target install
        if [ $link_dynamic = true ]; then
            cp $build_dir_release/libsgl.so install/lib/libsgl.so
        fi
    fi

    popd >/dev/null
fi

if $build_with_openvdb_support; then
    if [ ! -d "./openvdb" ]; then
        echo "------------------------"
        echo "  downloading OpenVDB   "
        echo "------------------------"
        if [ -d "./openvdb-src" ]; then
            rm -rf "./openvdb-src"
        fi
        git clone --recursive https://github.com/AcademySoftwareFoundation/openvdb.git openvdb-src
        mkdir -p openvdb-src/build
        pushd openvdb-src/build >/dev/null
        cmake .. ${params_gen[@]+"${params_gen[@]}"} -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/openvdb"
        make -j $(nproc)
        make install
        popd >/dev/null
    fi
    params+=(-DCMAKE_MODULE_PATH="${projectpath}/third_party/openvdb/lib/cmake/OpenVDB")
fi

if $use_custom_jsoncpp; then
    if [ ! -d "./jsoncpp" ]; then
        echo "------------------------"
        echo "  downloading JsonCpp   "
        echo "------------------------"
        if [ -d "./jsoncpp-src" ]; then
            rm -rf "./jsoncpp-src"
        fi
        git clone --recursive https://github.com/open-source-parsers/jsoncpp.git jsoncpp-src
        mkdir -p jsoncpp-src/build
        pushd jsoncpp-src/build >/dev/null
        cmake .. ${params_gen[@]+"${params_gen[@]}"} -DBUILD_STATIC_LIBS=OFF -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release -DJSONCPP_WITH_TESTS=OFF -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF \
        -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/jsoncpp"
        make -j $(nproc)
        make install
        popd >/dev/null
    fi
    params+=(-Djsoncpp_DIR="${projectpath}/third_party/jsoncpp/lib/cmake/jsoncpp")
fi

if $use_custom_openexr; then
    if [ ! -d "./openexr" ]; then
        echo "------------------------"
        echo "  downloading OpenEXR   "
        echo "------------------------"
        if [ -d "./openexr-src" ]; then
            rm -rf "./openexr-src"
        fi
        git clone --recursive https://github.com/AcademySoftwareFoundation/openexr openexr-src
        mkdir -p openexr-src/build
        pushd openexr-src/build >/dev/null
        cmake .. ${params_gen[@]+"${params_gen[@]}"}  -DCMAKE_BUILD_TYPE=Release \
        -DOPENEXR_INSTALL=ON -DOPENEXR_INSTALL_TOOLS=OFF \
        -DCMAKE_INSTALL_PREFIX="${projectpath}/third_party/openexr"
        make -j $(nproc)
        make install
        popd >/dev/null
    fi
    params+=(-DOpenEXR_DIR="${projectpath}/third_party/openexr/lib/cmake/OpenEXR")
fi

if $use_open_image_denoise; then
    oidn_version="2.3.0"
    if $use_msys; then
        oidn_folder_name="oidn-${oidn_version}.x64.windows"
        oidn_archive_name="${oidn_folder_name}.zip"
    elif $use_macos; then
        oidn_folder_name="oidn-${oidn_version}.${os_arch}.linux"
        oidn_archive_name="${oidn_folder_name}.tar.gz"
    else
        oidn_folder_name="oidn-${oidn_version}.${os_arch}.macos"
        oidn_archive_name="${oidn_folder_name}.tar.gz"
    fi
    if [ ! -d "./${oidn_folder_name}" ]; then
        echo "------------------------"
        echo "downloading OpenImageDenoise"
        echo "------------------------"
        wget "https://github.com/OpenImageDenoise/oidn/releases/download/v${oidn_version}/${oidn_archive_name}"
        if $use_msys; then
            unzip "${oidn_archive_name}"
        elif $use_macos; then
            tar -xvzf "${oidn_archive_name}"
        fi
    fi
    params+=(-DOpenImageDenoise_DIR="${projectpath}/third_party/${oidn_folder_name}/lib/cmake/OpenImageDenoise-${oidn_version}")
    if [ $use_msys = true ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${projectpath}/third_party/${oidn_folder_name}/bin"
    else
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${projectpath}/third_party/${oidn_folder_name}/lib"
    fi
fi

popd >/dev/null # back to project root

if [ $debug = true ]; then
    echo "------------------------"
    echo "  building in debug     "
    echo "------------------------"
else
    echo "------------------------"
    echo "  building in release   "
    echo "------------------------"
fi

if [ -f "./$build_dir/CMakeCache.txt" ]; then
    remove_build_cache=false
    if grep -q vcpkg_installed "./$build_dir/CMakeCache.txt"; then
        cache_uses_vcpkg=true
    else
        cache_uses_vcpkg=false
    fi
    if ([ $use_vcpkg = true ] && [ $cache_uses_vcpkg = false ]) || ([ $use_vcpkg = false ] && [ $cache_uses_vcpkg = true ]); then
        remove_build_cache=true
    fi
    if grep -q "USE_PRE_CXX11_ABI:BOOL=ON" "./$build_dir/CMakeCache.txt"; then
        cache_uses_pre_cxx11_abi=true
    else
        cache_uses_pre_cxx11_abi=false
    fi
    if ([ $use_pre_cxx11_abi = true ] && [ $cache_uses_pre_cxx11_abi = false ]) || ([ $use_pre_cxx11_abi = false ] && [ $cache_uses_pre_cxx11_abi = true ]); then
        remove_build_cache=true
    fi
    if [ remove_build_cache = true ]; then
        echo "Removing old application build cache..."
        if [ -d "./$build_dir_debug" ]; then
            rm -rf "./$build_dir_debug"
        fi
        if [ -d "./$build_dir_release" ]; then
            rm -rf "./$build_dir_release"
        fi
        if [ -d "./$destination_dir" ]; then
            rm -rf "./$destination_dir"
        fi
    fi
fi

mkdir -p $build_dir

echo "------------------------"
echo "      generating        "
echo "------------------------"
pushd $build_dir >/dev/null
cmake .. \
    -DCMAKE_BUILD_TYPE=$cmake_config \
    -Dsgl_DIR="$projectpath/third_party/sgl/install/lib/cmake/sgl/" \
    ${params_gen[@]+"${params_gen[@]}"} ${params_link[@]+"${params_link[@]}"} \
    ${params_vcpkg[@]+"${params_vcpkg[@]}"} ${params[@]+"${params[@]}"}
popd >/dev/null

echo "------------------------"
echo "      compiling         "
echo "------------------------"
if [ $use_macos = true ]; then
    cmake --build $build_dir --parallel $(sysctl -n hw.ncpu)
elif [ $use_vcpkg = true ]; then
    cmake --build $build_dir --parallel $(nproc)
else
    pushd "$build_dir" >/dev/null
    make -j $(nproc)
    popd >/dev/null
fi
if [ $use_pytorch = true ] && [ $install_module = true ]; then
    if [ $use_macos = true ] || [ $use_vcpkg = true ]; then
        cmake --build $build_dir --target install
    else
        pushd "$build_dir" >/dev/null
        make install
        popd >/dev/null
    fi
    if [ $debug = true ] ; then
        cp "./third_party/sgl/install/lib/libsgld.so" "$install_dir/modules"
    else
        cp "./third_party/sgl/install/lib/libsgl.so" "$install_dir/modules"
    fi
    if $build_with_openvdb_support; then
        cp $(ldd $build_dir/libvpt.so | grep libopenvdb | awk 'NF == 4 {print $3}; NF == 2 {print $1}') "$install_dir/modules"
    fi
    if $use_custom_openexr; then
        cp $(ldd $build_dir/libvpt.so | grep libOpenEXR | awk 'NF == 4 {print $3}; NF == 2 {print $1}') "$install_dir/modules"
        cp $(ldd $build_dir/libvpt.so | grep libIex | awk 'NF == 4 {print $3}; NF == 2 {print $1}') "$install_dir/modules"
        cp $(ldd $build_dir/libvpt.so | grep libIlmThread | awk 'NF == 4 {print $3}; NF == 2 {print $1}') "$install_dir/modules"
    fi
    if $use_open_image_denoise; then
        cp $(ldd $build_dir/libvpt.so | grep OpenImage | awk 'NF == 4 {print $3}; NF == 2 {print $1}') "$install_dir/modules"
    fi
    patchelf --set-rpath '$ORIGIN' "$install_dir/modules/libvpt.so"
fi

echo "------------------------"
echo "   copying new files    "
echo "------------------------"

# https://stackoverflow.com/questions/2829613/how-do-you-tell-if-a-string-contains-another-string-in-posix-sh
contains() {
    string="$1"
    substring="$2"
    if test "${string#*$substring}" != "$string"
    then
        return 0
    else
        return 1
    fi
}
startswith() {
    string="$1"
    prefix="$2"
    if test "${string#$prefix}" != "$string"
    then
        return 0
    else
        return 1
    fi
}

if $use_msys; then
    mkdir -p $destination_dir/bin

    # Copy sgl to the destination directory.
    if [ $debug = true ] ; then
        cp "./third_party/sgl/install/bin/libsgld.dll" "$destination_dir/bin"
    else
        cp "./third_party/sgl/install/bin/libsgl.dll" "$destination_dir/bin"
    fi

    # Copy the application to the destination directory.
    cp "$build_dir/CloudRendering.exe" "$destination_dir/bin"

    # Copy all dependencies of the application to the destination directory.
    ldd_output="$(ntldd -R $destination_dir/bin/CloudRendering.exe)"
    for library_abs in $ldd_output
    do
        library="$(cygpath "$library_abs")"
        if [[ $library == "$MSYSTEM_PREFIX"* ]] or [[ $library == "$projectpath"* ]];
        then
            cp "$library" "$destination_dir/bin"
        fi
        if [[ $library == libpython* ]];
        then
            tmp=${library#*lib}
            Python3_VERSION=${tmp%.dll}
        fi
    done
elif [ $use_macos = true ] && [ $use_vcpkg = true ]; then
    [ -d $destination_dir ] || mkdir $destination_dir
    rsync -a "$build_dir/CloudRendering.app/Contents/MacOS/CloudRendering" $destination_dir
elif [ $use_macos = true ] && [ $use_vcpkg = false ]; then
    brew_prefix="$(brew --prefix)"
    mkdir -p $destination_dir

    if [ -d "$destination_dir/CloudRendering.app" ]; then
        rm -rf "$destination_dir/CloudRendering.app"
    fi

    # Copy the application to the destination directory.
    cp -a "$build_dir/CloudRendering.app" "$destination_dir"

    # Copy sgl to the destination directory.
    if [ $debug = true ] ; then
        cp "./third_party/sgl/install/lib/libsgld.dylib" "$binaries_dest_dir"
    else
        cp "./third_party/sgl/install/lib/libsgl.dylib" "$binaries_dest_dir"
    fi

    # Copy all dependencies of the application and sgl to the destination directory.
    rsync -a "$VULKAN_SDK/lib/libMoltenVK.dylib" "$binaries_dest_dir"
    copy_dependencies_recursive() {
        local binary_path="$1"
        local binary_source_folder=$(dirname "$binary_path")
        local binary_name=$(basename "$binary_path")
        local binary_target_path="$binaries_dest_dir/$binary_name"
        if contains "$(file "$binary_target_path")" "dynamically linked shared library"; then
            install_name_tool -id "@executable_path/$binary_name" "$binary_target_path" &> /dev/null
        fi
        local otool_output="$(otool -L "$binary_path")"
        local otool_output=${otool_output#*$'\n'}
        while read -r line
        do
            local stringarray=($line)
            local library=${stringarray[0]}
            local library_name=$(basename "$library")
            local library_target_path="$binaries_dest_dir/$library_name"
            if ! startswith "$library" "@rpath/" \
                && ! startswith "$library" "@loader_path/" \
                && ! startswith "$library" "/System/Library/Frameworks/" \
                && ! startswith "$library" "/usr/lib/"
            then
                install_name_tool -change "$library" "@executable_path/$library_name" "$binary_target_path" &> /dev/null

                if [ ! -f "$library_target_path" ]; then
                    cp "$library" "$binaries_dest_dir"
                    copy_dependencies_recursive "$library"
                fi
            elif startswith "$library" "@rpath/"; then
                install_name_tool -change "$library" "@executable_path/$library_name" "$binary_target_path" &> /dev/null

                local rpath_grep_string="$(otool -l "$binary_target_path" | grep RPATH -A2)"
                local counter=0
                while read -r grep_rpath_line
                do
                    if [ $(( counter % 4 )) -eq 2 ]; then
                        local stringarray_grep_rpath_line=($grep_rpath_line)
                        local rpath=${stringarray_grep_rpath_line[1]}
                        if startswith "$rpath" "@loader_path"; then
                            rpath="${rpath/@loader_path/$binary_source_folder}"
                        fi
                        local library_rpath="${rpath}${library#"@rpath"}"

                        if [ -f "$library_rpath" ]; then
                            if [ ! -f "$library_target_path" ]; then
                                cp "$library_rpath" "$binaries_dest_dir"
                                copy_dependencies_recursive "$library_rpath"
                            fi
                            break
                        fi
                    fi
                    counter=$((counter + 1))
                done < <(echo "$rpath_grep_string")
            fi
        done < <(echo "$otool_output")
    }
    copy_dependencies_recursive "$build_dir/CloudRendering.app/Contents/MacOS/CloudRendering"
    if [ $debug = true ]; then
        copy_dependencies_recursive "./third_party/sgl/install/lib/libsgld.dylib"
    else
        copy_dependencies_recursive "./third_party/sgl/install/lib/libsgl.dylib"
    fi

    # Fix code signing for arm64.
    for filename in $binaries_dest_dir/*
    do
        if contains "$(file "$filename")" "arm64"; then
            codesign --force -s - "$filename" &> /dev/null
        fi
    done
${copy_dependencies_macos_post}
else
    mkdir -p $destination_dir/bin

    # Copy the application to the destination directory.
    rsync -a "$build_dir/CloudRendering" "$destination_dir/bin"

    # Copy all dependencies of the application to the destination directory.
    ldd_output="$(ldd $build_dir/CloudRendering)"

    library_blacklist=(
        "libOpenGL" "libGLdispatch" "libGL.so" "libGLX.so"
        "libwayland" "libffi." "libX" "libxcb" "libxkbcommon"
        "ld-linux" "libdl." "libutil." "libm." "libc." "libpthread." "libbsd." "librt."
    )
    if [ $use_vcpkg = true ]; then
        # We build with libstdc++.so and libgcc_s.so statically. If we were to ship them, libraries opened with dlopen will
        # use our, potentially older, versions. Then, we will get errors like "version `GLIBCXX_3.4.29' not found" when
        # the Vulkan loader attempts to load a Vulkan driver that was built with a never version of libstdc++.so.
        # I tried to solve this by using "patchelf --replace-needed" to directly link to the patch version of libstdc++.so,
        # but that made no difference whatsoever for dlopen.
        library_blacklist+=("libstdc++.so")
        library_blacklist+=("libgcc_s.so")
    fi
    for library in $ldd_output
    do
        if [[ $library != "/"* ]]; then
            continue
        fi
        is_blacklisted=false
        for blacklisted_library in ${library_blacklist[@]+"${library_blacklist[@]}"}; do
            if [[ "$library" == *"$blacklisted_library"* ]]; then
                is_blacklisted=true
                break
            fi
        done
        if [ $is_blacklisted = true ]; then
            continue
        fi
        # TODO: Add blacklist entries for pulseaudio and dependencies when not using vcpkg.
        #cp "$library" "$destination_dir/bin"
        #patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/$(basename "$library")"
        if [ $use_vcpkg = true ]; then
            cp "$library" "$destination_dir/bin"
            patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/$(basename "$library")"
        fi
    done
    patchelf --set-rpath '$ORIGIN' "$destination_dir/bin/CloudRendering"
fi


# Copy the docs to the destination directory.
cp "README.md" "$destination_dir"
if [ ! -d "$destination_dir/LICENSE" ]; then
    mkdir -p "$destination_dir/LICENSE"
    cp -r "docs/license-libraries/." "$destination_dir/LICENSE/"
    cp -r "LICENSE" "$destination_dir/LICENSE/LICENSE-cloudrendering.txt"
    cp -r "submodules/IsosurfaceCpp/LICENSE" "$destination_dir/LICENSE/graphics/LICENSE-isosurfacecpp.txt"
fi
if [ ! -d "$destination_dir/docs" ]; then
    cp -r "docs" "$destination_dir"
fi

# Create a run script.
if $use_msys; then
    printf "@echo off\npushd %%~dp0\npushd bin\nstart \"\" CloudRendering.exe\n" > "$destination_dir/run.bat"
elif $use_macos; then
    printf "#!/bin/sh\npushd \"\$(dirname \"\$0\")\" >/dev/null\n./CloudRendering.app/Contents/MacOS/CloudRendering\npopd\n" > "$destination_dir/run.sh"
    chmod +x "$destination_dir/run.sh"
else
    printf "#!/bin/bash\npushd \"\$(dirname \"\$0\")/bin\" >/dev/null\n./CloudRendering\npopd\n" > "$destination_dir/run.sh"
    chmod +x "$destination_dir/run.sh"
fi



# Run the program as the last step.
echo ""
echo "All done!"
pushd $build_dir >/dev/null

if $use_msys; then
    if [[ -z "${PATH+x}" ]]; then
        export PATH="${projectpath}/third_party/sgl/install/bin"
    elif [[ ! "${PATH}" == *"${projectpath}/third_party/sgl/install/bin"* ]]; then
        export PATH="${projectpath}/third_party/sgl/install/bin:$PATH"
    fi
elif $use_macos; then
    if [ -z "${DYLD_LIBRARY_PATH+x}" ]; then
        export DYLD_LIBRARY_PATH="${projectpath}/third_party/sgl/install/lib"
    elif contains "${DYLD_LIBRARY_PATH}" "${projectpath}/third_party/sgl/install/lib"; then
        export DYLD_LIBRARY_PATH="DYLD_LIBRARY_PATH:${projectpath}/third_party/sgl/install/lib"
    fi
else
  if [[ -z "${LD_LIBRARY_PATH+x}" ]]; then
      export LD_LIBRARY_PATH="${projectpath}/third_party/sgl/install/lib"
  elif [[ ! "${LD_LIBRARY_PATH}" == *"${projectpath}/third_party/sgl/install/lib"* ]]; then
      export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${projectpath}/third_party/sgl/install/lib"
  fi
fi

if [ $run_program = true ] && [ $use_macos = false ]; then
    ./CloudRendering ${params_run[@]+"${params_run[@]}"}
elif [ $run_program = true ] && [ $use_macos = true ]; then
    #open ./CloudRendering.app
    #open ./CloudRendering.app --args --perf
    ./CloudRendering.app/Contents/MacOS/CloudRendering ${params_run[@]+"${params_run[@]}"}
fi

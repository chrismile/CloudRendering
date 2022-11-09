# Volumetric Path Tracing Renderer for Clouds

This repository contains a volumetric path tracing renderer written in C++ using Vulkan.

![Teaser image of a cloud rendered using volumetric path tracing.](https://chrismile.net/github/cloud-rendering/vpt-cloud.png)


## Building and running the programm

### Linux

There are two ways to build the program on Linux systems.
- Using the system package manager to install all dependencies (tested: apt on Ubuntu, pacman on Arch Linux).
- Using [vcpkg](https://github.com/microsoft/vcpkg) to install all dependencies.

In the project root directory, two scripts `build-linux.sh` and `build-linux-vcpkg.sh` can be found. The former uses the
system package manager to install all dependencies, while the latter uses vcpkg. The build scripts will also launch the
program after successfully building it. If you wish to build the program manually, instructions can be found in the
directory `docs/compilation`.

Below, more information concerning different Linux distributions tested can be found.

#### Arch Linux

Arch Linux and its derivative Manjaro are fully supported using both build modes (package manager and vcpkg).

The Vulkan SDK will be automatically installed using the package manager `pacman` when using the scripts.

#### Ubuntu 18.04, 20.04 & 22.04

Ubuntu 20.04 and 22.04 are fully supported.

The Vulkan SDK will be automatically installed using the official PPA.

Please note that Ubuntu 18.04 is only partially supported. It ships an old version of CMake, which causes the build
process using vcpkg to fail if not updating CMake manually beforehand. Also, an old version of GLEW in the package
sources causes the Vulkan interoperability support in sgl to be disabled regardless of whether the Vulkan SDK is
installed if the system packages are used.

#### Other Linux Distributions

If you are using a different Linux distribution and face difficulties when building the program, please feel free to
open a [bug report](https://github.com/chrismile/CloudRendering/issues).


### Windows

There are two ways to build the program on Windows.
- Using [vcpkg](https://github.com/microsoft/vcpkg) to install all dependencies. The program can then be compiled using
  [Microsoft Visual Studio](https://visualstudio.microsoft.com/vs/).
- Using [MSYS2](https://www.msys2.org/) to install all dependencies and compile the program using MinGW.

In the project folder, a script called `build-windows.bat` can be found automating this build process using vcpkg and
Visual Studio. It is recommended to run the script using the `Developer PowerShell for VS 2022` (or VS 2019 depending on
your Visual Studio version). The build script will also launch the program after successfully building it.
Building the program is regularly tested on Windows 10 and 11 with Microsoft Visual Studio 2019 and 2022.

Please note that the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home#windows) needs to be installed before starting the
build process.

A script `build-windows-msys2.bat` is also available to build the program using MSYS2/MinGW alternatively to using
Microsoft Visual Studio.

If you wish to build the program manually using Visual Studio and vcpkg, or using MSYS2, instructions can be found in
the directory `docs/compilation`.


### macOS

There are two ways to build the program on macOS.
- Using [Homebrew](https://brew.sh/) to install the dependencies and compile the program using LLVM/Clang (recommended).
- Using [vcpkg](https://github.com/microsoft/vcpkg) to install the dependencies and compile the program using
  LLVM/Clang.

In the project root directory, two scripts `build-macos-vcpkg.sh` and `build-macos-brew.sh` can be found.
As macOS does not natively support Vulkan, MoltenVK, a Vulkan wrapper based on Apple's Metal API, is utilized.
Installing it via the scripts requires admin rights. MoltenVK can also be installed manually from
[the website](https://vulkan.lunarg.com/sdk/home#mac).

Notes:
- I rented Apple hardware for a few days once for testing that running the program works on macOS.
  As I do not regularly have access to a real system running macOS, it is only tested that the program can compile in a
  CI pipeline build script on an x86_64 macOS virtual machine provided by GitHub Actions. So please note that it is not
  guaranteed that the program will continue working correctly on macOS indefinitely due to the lack of regular testing.
- To enable high DPI support, the program needs to be run from an app bundle. This happens automatically when the script
  `build-macos-brew.sh` has finished building the program. Please note that the app bundle only contains the Info.plist
  file necessary for high DPI support and is currently not yet redistributable. If you want to help with improving the
  macOS app bundle support for this project by contributing development time, please feel free to contact me.


### Unit Tests

Unit tests using the [GoogleTest framework](https://github.com/google/googletest) can be built by passing the argument
`-DUSE_GTEST=On` to CMake.
These unit tests can also be run using software rendering via [SwiftShader](https://github.com/google/swiftshader).


### PyTorch Module (Work in Progress)

A [PyTorch](https://pytorch.org/) module can be built by passing `-DBUILD_PYTORCH_MODULE=On` to CMake.

It provides the function `initialize`, `cleanup` and `render_frame` and works both with CPU tensors and CUDA tensors.
To use this module, the dependency sgl must have been built using CUDA interoperability support (this should happen
automatically when CUDA is detected on the system).

The path to where the module should be installed can be specified using `-DCMAKE_INSTALL_PREFIX=/path/to/dir`.
If TorchLib does not lie on a standard path, the directory where the CMake config files of TorchLib lie must be
specified using, e.g.:

```
-DCMAKE_PREFIX_PATH=~/miniconda3/envs/cloud_rendering/lib/python3.8/site-packages/torch/share/cmake
```


Additionally, if using the module on Linux, PyTorch must have been build using the C++11 ABI.
This is not the case for pre-built PyTorch packages as of 2022-02-14.
The command `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` can be used to check whether PyTorch was
built using the C++11 ABI.

If necessary, PyTorch can be built manually using the commands below (assuming the CUDA Toolkit version 11.5 and
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) are installed on the system).

```shell
sudo apt install g++ git libgflags-dev libgoogle-glog-dev libopenmpi-dev protobuf-compiler python3 python3-pip \
python3-setuptools python3-yaml wget intel-mkl libcudnn8-dev

. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create --name cloud_rendering python=3.8
conda activate cloud_rendering

conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing typing_extensions future \
six requests dataclasses
conda install -c pytorch magma-cuda115

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
export USE_MKLDNN=1
python setup.py install
```

HINT: In case the CUDA Toolkit is not found, the build process might just continue without building CUDA support.
Assuming the CUDA Toolkit was installed to `/usr/local/cuda-11.5` using the manual NVIDIA CUDA Toolkit installer, the
following lines might need to be added to `~/.profile` in order for PyTorch to find the installed CUDA version:

```shell
export CPATH=/usr/local/cuda-11.5/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.5/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-11.5/bin:$PATH
```

HINT 2: On Ubuntu 22.04 with Python 3.9 installed via Conda, a problem one user noticed was that GLIBCXX_3.4.30 was
used for building the PyTorch module using the system libstdc++, but the Conda libstdc++ did not support GLIBCXX_3.4.30.
The problem could be fixed by installing a newer version of libstdc++ in the Conda environment using the commands below.

```shell
conda install -c conda-forge libgcc-ng libstdcxx-ng
```

It is planned to also add support for the PyTorch Vulkan backend once the PyTorch Vulkan code base has sufficiently
matured. As of 2022-02-14, there are still some problems building PyTorch with Vulkan support on x86_64 Linux.


## How to add new data sets

Under `Data/CloudDataSets/datasets.json`, loadable data sets can be specified. Additionally, the user can also open
arbitrary data sets using a file explorer via "File > Open Dataset..." (or using Ctrl+O).

Below, an example for a `Data/CloudDataSets/datasets.json` file can be found.

```json
{
    "datasets": [
        { "name" : "Sphere (64x64x64)", "filename": "sphere_64x64x64.xyz" },
        { "name" : "Bunny", "filename": "bunny.nvdb" }
    ]
}
```

These files then appear with their specified name in the menu "File > Datasets". All paths must be specified relative to
the folder `Data/CloudDataSets/` (unless they are global, like `C:/path/file.dat` or `/path/file.dat`).

Supported formats currently are:
- .xyz files, which consist of a header of 3x float (grid size sx, sy, sz) and 3x double (voxel size vx, vy, vz)
  followed by sx * sy * sz floating point values storing the density values stored in the dense Cartesian grid.
- .nvdb files using the [NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb/nanovdb)
  format, which stores sparse voxel grids.


## Supported Rendering Modes

Below, a list of supported rendering modes can be found.

- Delta tracking and spectral delta tracking.

  E. Woodcock, T. Murphy, P.J. Hemmings, AND T.C. Longworth. Techniques used in the GEM code for Monte Carlo neutronics
  calculations in reactors and other systems of complex geometry. In Applications of Computing Methods to Reactor
  Problems, Argonne National Laboratory, 1965.

- Ratio tracking and residual ratio tracking (residual ratio tracking is still work in progress).

  J. Novák, A. Selle, and W. Jarosz. Residual ratio tracking for estimating attenuation in participating media.
  ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia) , 33(6), Nov. 2014.

- Decomposition tracking.

  P. Kutz, R. Habel, Y. K. Li, and J. Novák. Spectral and decomposition tracking for rendering heterogeneous volumes.
  ACM Trans. Graph., 36(4), Jul. 2017.

- Support for sparse grids using [NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb/tree/master/nanovdb/nanovdb).

  K. Museth. Nanovdb: A GPU-friendly and portable VDB data structure for real-time rendering and simulation.
  In ACM SIGGRAPH 2021 Talks, SIGGRAPH '21, New York, NY, USA, 2021. Association for Computing Machinery.


## How to report bugs

When [reporting a bug](https://github.com/chrismile/CloudRendering/issues), please also attach the logfile generated by this
program. Below, the location of the logfile on different operating systems can be found.

- Linux: `~/.config/cloud-rendering/Logfile.html`
- Windows: `C:/Users/<USER>/AppData/Roaming/CloudRendering/Logfile.html`

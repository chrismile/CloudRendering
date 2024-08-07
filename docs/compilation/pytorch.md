## Compile PyTorch Manually

If necessary, PyTorch can be built manually using the commands below (assuming the CUDA Toolkit version 12.4 and
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) are installed on the system).
`cudnn` and `cudnn-cuda-12` are not available by default, and the repository may need to be added from the
[cuDNN webpage](https://developer.nvidia.com/cudnn). On other operating systems than Ubuntu, it may be necessary to
follow the manual installation instructions on the webpage.

IMPORTANT: `python setup.py install` (the last command below) may use a lot of memory, depending on the number of
available CPU threads. `MAX_JOBS=4` can be prepended to reduce the number of build threads if this causes problems.

IMPORTANT: `python setup.py install` (the last command below) may use a lot of memory, depending on the number of
available CPU threads. `MAX_JOBS=4` can be prepended to reduce the number of build threads if this causes problems.

```shell
sudo apt install g++ git libgflags-dev libgoogle-glog-dev libopenmpi-dev protobuf-compiler python3 python3-pip \
python3-setuptools python3-yaml wget intel-mkl cudnn cudnn-cuda-12

. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create --name vpt python=3.12
conda activate vpt

conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing typing-extensions
conda install -c pytorch magma-cuda124

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# Optional: Use a stable version of PyTorch
#git checkout v2.3.1
#git submodule sync
#git submodule update --init --recursive
# Optional: Build for different GPU architectures.
#export TORCH_CUDA_ARCH_LIST="6.1 7.5 8.6"
#export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"
export USE_MKLDNN=1
python setup.py install
```

HINT: In case the CUDA Toolkit is not found, the build process might just continue without building CUDA support.
Assuming the CUDA Toolkit was installed to `/usr/local/cuda-12.4` using the manual NVIDIA CUDA Toolkit installer, the
following lines might need to be added to `~/.profile` in order for PyTorch to find the installed CUDA version:

```shell
export CPATH=/usr/local/cuda-12.4/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.4/bin:$PATH
```

HINT 2: On Ubuntu 22.04 with Python 3.9 installed via Conda, a problem one user noticed was that GLIBCXX_3.4.30 was
used for building the PyTorch module using the system libstdc++, but the Conda libstdc++ did not support GLIBCXX_3.4.30.
The problem could be fixed by installing a newer version of libstdc++ in the Conda environment using the commands below.

```shell
conda install -c conda-forge libgcc-ng libstdcxx-ng
```

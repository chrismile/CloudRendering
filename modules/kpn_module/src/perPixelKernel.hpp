#ifndef PER_PIXEL_KERNEL_HPP
#define PER_PIXEL_KERNEL_HPP

#include <torch/script.h>
#include <torch/types.h>


torch::Tensor perPixelKernelForward(torch::Tensor image, torch::Tensor weights, int64_t kernelSize);
torch::Tensor perPixelKernelCuda(torch::Tensor image, torch::Tensor weights, int64_t kernelSize);
torch::Tensor perPixelKernelForwardB(torch::Tensor image, torch::Tensor weights, int64_t kernelSize);
torch::Tensor perPixelKernelCudaB(torch::Tensor image, torch::Tensor weights, int64_t kernelSize);
#endif //PER_PIXEL_KERNEL_HPP

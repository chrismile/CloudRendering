#include "perPixelKernel.hpp"

TORCH_LIBRARY(per_pixel_kernel, m) {
    m.def("per_pixel_kernel::forward", perPixelKernelForward);
    m.def("per_pixel_kernel::forwardB", perPixelKernelForwardB);
}

torch::Tensor perPixelKernelForward(torch::Tensor image, torch::Tensor weights, int64_t kernelSize) {
    if (image.device().is_cuda()) {
        return perPixelKernelCuda(image, weights, kernelSize);
    } else {
        throw std::runtime_error("Error in perPixelKernelForward: Unsupported device.");
    }
}

torch::Tensor perPixelKernelForwardB(torch::Tensor image, torch::Tensor weights, int64_t kernelSize) {
    if (image.device().is_cuda()) {
        return perPixelKernelCudaB(image, weights, kernelSize);
    } else {
        throw std::runtime_error("Error in perPixelKernelForward: Unsupported device.");
    }
}
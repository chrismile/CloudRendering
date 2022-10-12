#include "perPixelKernel.hpp"

TORCH_LIBRARY(per_pixel_kernel, m) {
    m.def("per_pixel_kernel::forward", perPixelKernelForward);
}

torch::Tensor perPixelKernelForward(torch::Tensor image, torch::Tensor weights, int kernelSize) {
    if (image.device().is_cuda()) {
        return perPixelKernelCuda(image, weights, kernelSize);
    } else {
        throw std::runtime_error("Error in perPixelKernelForward: Unsupported device.");
    }
}

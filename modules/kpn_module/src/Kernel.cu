#include <c10/cuda/CUDAStream.h>
#include "perPixelKernel.hpp"

#define MAX_CHANNELS (4)


inline int iceil(int x, int y) {
    return 1 + ((x - 1) / y);
}

__global__ void perPixelKernelForwardKernel(
        float* image,
        float* weights,
        float* output,
        const int kernelSize, const int h, const int w, const int numChannels,
        const int numBatches
        ) {

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.z + blockIdx.z * blockDim.z;
    //const int channel = threadIdx.z + blockIdx.z * blockDim.z;
    //float result = 0;

    float result[MAX_CHANNELS];
    for (int c = 0; c < numChannels; c++){
        result[c] = 0;
    }

    if (x >= w || y >= h) {
        return;
    }
    int weightIdx = 0;
    for (int dy = -kernelSize/2; dy <= kernelSize/2; dy++){
        int ky = y + dy;
        for (int dx = -kernelSize/2; dx <= kernelSize/2; dx++){
            int kx = x + dx;
            if (kx >= 0 && kx < w && ky >= 0 && ky < h) {
                float weight = weights[batch * w * h * kernelSize * kernelSize + weightIdx * w * h + y * w + x];
                for (int c = 0; c < numChannels; c++) {
                    result[c] += weight * image[batch * w * h * numChannels + c * w * h + ky * w + kx];
                }
            }
            weightIdx++;
        }
    }
    for (int c = 0; c < numChannels; c++){
        output[batch * w * h * numChannels+ c * w * h + y * w + x] = result[c];
    }
}

torch::Tensor perPixelKernelCuda(torch::Tensor image, torch::Tensor weights, int64_t kernelSize){
    if (image.sizes().size() != 4) {
        throw std::runtime_error("Error in perPixelKernelCuda: image.sizes().size() != 4.");
    }
    if (weights.sizes().size() != 4) {
        throw std::runtime_error("Error in perPixelKernelCuda: weights.sizes().size() != 4.");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int64_t N = image.size(0);
    const int64_t C = image.size(1);
    const int64_t H = image.size(2);
    const int64_t W = image.size(3);

    if (H != weights.size(2) || W != weights.size((3))) {
        throw std::runtime_error("Error, image and weight sizes dont match");
    }
    if (C > MAX_CHANNELS) {
        throw std::runtime_error("Error, must have <= 4 channels");
    }

    torch::Tensor result = torch::zeros_like(image);
    auto imageAccessor = image.packed_accessor32<float, 4>();
    auto weightsAccessor = weights.packed_accessor32<float, 4>();
    auto resultAccessor = result.packed_accessor32<float, 4>();

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(iceil(W, blockDim.x), iceil(H, blockDim.y), N);

    perPixelKernelForwardKernel<<<gridDim, blockDim, 0, stream>>> (
            (float*)image.data_ptr(), (float*)weights.data_ptr(), (float*)result.data_ptr(),
            kernelSize, H, W, C, N
    );

    return result;
}
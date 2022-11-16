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
        const int kernelSize,
        const int h, const int w, // Size of output (min of image and kernel size)
        const int img_h, const int img_w,   // size of img
        const int weight_h, const int weight_w, // size of weights
        const int numChannels, const int numBatches
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
                float weight = weights[batch * weight_w * weight_h * kernelSize * kernelSize + weightIdx * weight_w * weight_h + y * weight_w + x];
                for (int c = 0; c < numChannels; c++) {
                    result[c] += weight * image[batch * img_w * img_h * numChannels + c * img_w * img_h + ky * img_w + kx];
                }
            }
            weightIdx++;
        }
    }
    for (int c = 0; c < numChannels; c++){
        output[batch * w * h * numChannels+ c * w * h + y * w + x] = result[c];
    }
    /*for (int c = 0; c < numChannels; c++){
        output[batch * w * h * numChannels+ c * w * h + y * w + x] = image[batch * img_w * img_h * numChannels + c * img_w * img_h + y * img_w + x];
    }*/
}

torch::Tensor perPixelKernelCuda(torch::Tensor image, torch::Tensor weights, int64_t kernelSize){
    if (image.sizes().size() != 4) {
        throw std::runtime_error("Error in perPixelKernelCuda: image.sizes().size() != 4.");
    }
    if (weights.sizes().size() != 4) {
        throw std::runtime_error("Error in perPixelKernelCuda: weights.sizes().size() != 4.");
    }
    if (!image.is_contiguous()) {
        image = image.contiguous();
    }
    if (!weights.is_contiguous()) {
        weights = weights.contiguous();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int64_t imgN = image.size(0);
    const int64_t imgC = image.size(1);
    const int64_t imgH = image.size(2);
    const int64_t imgW = image.size(3);

    const int64_t weightH = weights.size(2);
    const int64_t weightW = weights.size(3);

    const int64_t h = std::min(imgH, weightH);
    const int64_t w = std::min(imgW, weightW);

    if (imgC > MAX_CHANNELS) {
        throw std::runtime_error("Error, must have <= 4 channels");
    }

    torch::Tensor result = torch::zeros({imgN, imgC, h, w}, torch::TensorOptions().dtype(torch::kFloat32).device(image.device()));
    //torch::Tensor result = torch::zeros_like(image);
    auto imageAccessor = image.packed_accessor32<float, 4>();
    auto weightsAccessor = weights.packed_accessor32<float, 4>();
    auto resultAccessor = result.packed_accessor32<float, 4>();

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(iceil(w, blockDim.x), iceil(h, blockDim.y), imgN);

    perPixelKernelForwardKernel<<<gridDim, blockDim, 0, stream>>> (
            (float*)image.data_ptr(),
            (float*)weights.data_ptr(),
            (float*)result.data_ptr(),
            kernelSize,
            h, w, imgH, imgW, weightH, weightW,
            imgC, imgN
    );

    return result;
}


__global__ void getInvIterBaseKernel(
        float* hi_res_x,
        float* low_res_pred,
        float* low_res_x,
        float* prev_pred,
        float* weights,
        float* output,
        const int kernelSize,
        const int h, const int w, // Size of output (min of image and kernel size)
        const int img_h, const int img_w,   // size of img
        const int lo_res_h, const int lo_res_w, // size of low res images
        const int weight_h, const int weight_w, // size of weights
        const int numChannels, const int numBatches
) {

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int lx = x / 2;
    const int ly = y / 2;

    const int batch = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= w || y >= h || lx >= lo_res_w || ly >= lo_res_h) {
        return;
    }
    for (int c = 0; c < numChannels; c++){

        float lf_weight = weights[batch * weight_w * weight_h * kernelSize * kernelSize + 25 * weight_w * weight_h + y * weight_w + x];
        float hf_weight = weights[batch * weight_w * weight_h * kernelSize * kernelSize + 26 * weight_w * weight_h + y * weight_w + x];
        float prev_weight = weights[batch * weight_w * weight_h * kernelSize * kernelSize + 27 * weight_w * weight_h + y * weight_w + x];
        lf_weight = 1.0f / (1.0f + expf(-lf_weight));
        hf_weight = 1.0f / (1.0f + expf(-hf_weight));
        prev_weight = 1.0f / (1.0f + expf(-prev_weight));

        float x_hi = hi_res_x[batch * img_w * img_h * numChannels + c * img_w * img_h + y * img_w + x];
        float x_lo = low_res_x[batch * lo_res_w * lo_res_h * numChannels + c * lo_res_w * lo_res_h + ly * lo_res_w + lx];
        float pred_lo = low_res_pred[batch * lo_res_w * lo_res_h * numChannels + c * lo_res_w * lo_res_h + ly * lo_res_w + lx];
        float prev = prev_pred[batch * img_w * img_h * numChannels + c * img_w * img_h + y * img_w + x];

        float lf = lf_weight * pred_lo + (1.0f - lf_weight) * x_lo;
        lf = lf * (1.0 - prev_weight) + prev * prev_weight;
        float base = hf_weight * (x_hi - x_lo + lf) + (1.0f - hf_weight) * lf;

        output[batch * w * h * numChannels+ c * w * h + y * w + x] = base;
    }
}

torch::Tensor getInvIterBaseCuda(torch::Tensor x, torch::Tensor low_res_pred, torch::Tensor low_res_x, torch::Tensor prev, torch::Tensor fused_weights, int kernelSize){
    if (x.sizes().size() != 4) {
        throw std::runtime_error("Error in perPixelKernelCuda: image.sizes().size() != 4.");
    }
    if (fused_weights.sizes().size() != 4) {
        throw std::runtime_error("Error in perPixelKernelCuda: weights.sizes().size() != 4.");
    }
    std::cout << "getting base" << std::endl;
    std::cout << "x: " << x.sizes() << std::endl;
    std::cout << "pred: " << low_res_pred.sizes() << std::endl;
    std::cout << "x_lo: " << low_res_x.sizes() << std::endl;

    if (!x.is_contiguous()) {
        std::cout << "x not contiguous" << std::endl;
        x = x.contiguous();
    }
    if (!low_res_pred.is_contiguous()) {
        std::cout << "lo_pred not contiguous" << std::endl;
        low_res_pred = low_res_pred.contiguous();
    }
    if (!low_res_x.is_contiguous()) {
        std::cout << "lo_x not contiguous" << std::endl;
        low_res_x = low_res_x.contiguous();
    }
    if (!fused_weights.is_contiguous()) {
        std::cout << "fused_we not contiguous" << std::endl;
        fused_weights = fused_weights.contiguous();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int64_t imgN = x.size(0);
    const int64_t imgC = x.size(1);
    const int64_t imgH = x.size(2);
    const int64_t imgW = x.size(3);

    if (prev.size(2) != imgH || prev.size(3) != imgW){
        std::cerr << "previous size mismatched" << std::endl;
    }

    const int64_t weightH = fused_weights.size(2);
    const int64_t weightW = fused_weights.size(3);

    const int64_t lowH = low_res_x.size(2);
    const int64_t lowW = low_res_x.size(3);

    const int64_t h = std::min(imgH, weightH);
    const int64_t w = std::min(imgW, weightW);

    if (imgC > MAX_CHANNELS) {
        throw std::runtime_error("Error, must have <= 4 channels");
    }

    torch::Tensor result = torch::zeros({imgN, imgC, h, w}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    //torch::Tensor result = torch::zeros_like(image);

    dim3 blockDim(16, 16, 1);
    dim3 gridDim(iceil(w, blockDim.x), iceil(h, blockDim.y), imgN);

    getInvIterBaseKernel<<<gridDim, blockDim, 0, stream>>> (
            (float*)x.data_ptr(),
            (float*)low_res_pred.data_ptr(),
            (float*)low_res_x.data_ptr(),
            (float*)prev.data_ptr(),
            (float*)fused_weights.data_ptr(),
            (float*)result.data_ptr(),
            kernelSize,
            h, w, imgH, imgW, lowH, lowW, weightH, weightW,
            imgC, imgN
    );

    return result;
}
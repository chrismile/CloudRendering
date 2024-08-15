/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Math/Math.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Utils/InteropCustom.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <Graphics/Vulkan/Render/Passes/BlitRenderPass.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "OpenImageDenoiseDenoiser.hpp"

class AlphaBlitPass : public sgl::vk::ComputePass {
public:
    explicit AlphaBlitPass(sgl::vk::Renderer* renderer);
    void setInputImage(const sgl::vk::ImageViewPtr& _inputImage);
    void setOutputImage(const sgl::vk::ImageViewPtr& _outputImage);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const int BLOCK_SIZE = 16;
    sgl::vk::ImageViewPtr inputImage;
    sgl::vk::ImageViewPtr outputImage;
};

AlphaBlitPass::AlphaBlitPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
}

void AlphaBlitPass::setInputImage(const sgl::vk::ImageViewPtr& _inputImage) {
    inputImage = _inputImage;
    if (computeData) {
        computeData->setStaticImageView(inputImage, "inputImage");
    }
}

void AlphaBlitPass::setOutputImage(const sgl::vk::ImageViewPtr& _outputImage) {
    outputImage = _outputImage;
    if (computeData) {
        computeData->setStaticImageView(outputImage, "outputImage");
    }
}

void AlphaBlitPass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "AlphaBlit.Compute" }, preprocessorDefines);
}

void AlphaBlitPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticImageView(inputImage, "inputImage");
    computeData->setStaticImageView(outputImage, "outputImage");
}

void AlphaBlitPass::_render() {
    auto width = int(inputImage->getImage()->getImageSettings().width);
    auto height = int(inputImage->getImage()->getImageSettings().height);
    renderer->dispatch(
            computeData,
            sgl::iceil(width, BLOCK_SIZE), sgl::iceil(height, BLOCK_SIZE), 1);
}



OpenImageDenoiseDenoiser::OpenImageDenoiseDenoiser(sgl::vk::Renderer* renderer) : renderer(renderer) {
    alphaBlitPass = std::make_shared<AlphaBlitPass>(renderer);
    _createDenoiser();
}

OpenImageDenoiseDenoiser::~OpenImageDenoiseDenoiser() {
    _freeBuffers();
    _freeDenoiser();
}


void OpenImageDenoiseDenoiser::_createDenoiser() {
    _freeBuffers();
    _freeDenoiser();

    if (deviceType == OIDNDeviceTypeCustom::GPU_UUID) {
        auto* device = renderer->getDevice();
        const VkPhysicalDeviceIDProperties& deviceIdProperties = device->getDeviceIDProperties();
        oidnDevice = oidnNewDeviceByUUID(deviceIdProperties.deviceUUID);
    } else {
        oidnDevice = oidnNewDevice(OIDNDeviceType(std::max(int(deviceType) - 1, 0)));
    }
    oidnCommitDevice(oidnDevice);

    // TODO: If support for manual CUDA is added, use:
    //oidnNewCUDADevice(deviceIds, streams, 1);
    // This has the advantage that we do not need to wait for added semaphore import support.
}

void OpenImageDenoiseDenoiser::_freeDenoiser() {
    if (oidnDevice) {
        oidnReleaseDevice(oidnDevice);
        oidnDevice = {};
    }
}


void OpenImageDenoiseDenoiser::_createBuffers() {
    oidnFilter = oidnNewFilter(oidnDevice, "RT");

    auto externalMemoryTypes = (OIDNExternalMemoryTypeFlag)oidnGetDeviceInt(oidnDevice, "externalMemoryTypes");
    supportsMemoryImport = false;
#ifdef __APPLE__
    supportsMemoryImport = true;
    oidnNewSharedBufferFromMetal(device, buffer->getMetalBufferId()));
#elif defined(_WIN32)
    if ((externalMemoryTypes & OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32) != 0) {
        supportsMemoryImport = true;
    }
#else
    if ((externalMemoryTypes & OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD) != 0) {
        supportsMemoryImport = true;
    }
#endif

    auto* device = renderer->getDevice();
    if (supportsMemoryImport) {
        auto byteSizeColor = inputWidth * inputHeight * 4 * sizeof(float);

        sgl::vk::BufferSettings bufferSettings{};
        bufferSettings.sizeInBytes = byteSizeColor;
        bufferSettings.usage =
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_ONLY;
#ifndef __APPLE__
        bufferSettings.exportMemory = true;
        bufferSettings.useDedicatedAllocationForExportedMemory = true;
#endif
        inputImageBufferVk = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);
        outputImageBufferVk = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);
#ifndef __APPLE__
        inputImageBufferInterop = std::make_shared<sgl::vk::BufferCustomInteropVk>(inputImageBufferVk);
        outputImageBufferInterop = std::make_shared<sgl::vk::BufferCustomInteropVk>(outputImageBufferVk);
#endif
#if defined(_WIN32)
        oidnInputColorBuffer = oidnNewSharedBufferFromWin32Handle(
                oidnDevice, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                inputImageBufferInterop->getHandle(), nullptr, byteSizeColor);
        oidnOutputColorBuffer = oidnNewSharedBufferFromWin32Handle(
                oidnDevice, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
                outputImageBufferInterop->getHandle(), nullptr, byteSizeColor);
#elif !defined(__APPLE__)
        oidnInputColorBuffer = oidnNewSharedBufferFromFD(
                oidnDevice, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
                inputImageBufferInterop->getFileDescriptor(), byteSizeColor);
        oidnOutputColorBuffer = oidnNewSharedBufferFromFD(
                oidnDevice, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD,
                outputImageBufferInterop->getFileDescriptor(), byteSizeColor);
#endif
    } else {
        sgl::vk::BufferSettings bufferSettings{};
        bufferSettings.sizeInBytes = inputWidth * inputHeight * 4 * sizeof(float);

        bufferSettings.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_TO_CPU;
        inputImageBufferVk = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);
        bufferSettings.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferSettings.memoryUsage = VMA_MEMORY_USAGE_CPU_TO_GPU;
        outputImageBufferVk = std::make_shared<sgl::vk::Buffer>(device, bufferSettings);

        oidnInputColorBuffer = oidnNewBuffer(oidnDevice, inputWidth * inputHeight * 4 * sizeof(float));
        oidnOutputColorBuffer = oidnNewBuffer(oidnDevice, inputWidth * inputHeight * 4 * sizeof(float));
    }

    // TODO - For manual CUDA:
    //oidnNewSharedBuffer(device, devPtr, byteSize);

    // TODO: FLOAT3, FLOAT4
    oidnSetFilterImage(
            oidnFilter, "color",  oidnInputColorBuffer, OIDN_FORMAT_FLOAT3, inputWidth, inputHeight,
            0, 4 * sizeof(float), 0);
    oidnSetFilterImage(
            oidnFilter, "output", oidnOutputColorBuffer, OIDN_FORMAT_FLOAT3, inputWidth, inputHeight,
            0, 4 * sizeof(float), 0);

    oidnSetFilterBool(oidnFilter, "hdr", true);
    oidnCommitFilter(oidnFilter);
}

void OpenImageDenoiseDenoiser::_freeBuffers() {
    if (oidnInputColorBuffer) {
        oidnReleaseBuffer(oidnInputColorBuffer);
        oidnInputColorBuffer = {};
    }
    if (oidnOutputColorBuffer) {
        oidnReleaseBuffer(oidnOutputColorBuffer);
        oidnOutputColorBuffer = {};
    }
    if (oidnFilter) {
        oidnReleaseFilter(oidnFilter);
        oidnFilter = {};
    }
    inputImageBufferInterop = {};
    outputImageBufferInterop = {};
    inputImageBufferVk = {};
    outputImageBufferVk = {};
}

void OpenImageDenoiseDenoiser::setOutputImage(sgl::vk::ImageViewPtr& outputImage) {
    outputImageVulkan = outputImage;
}

void OpenImageDenoiseDenoiser::setFeatureMap(FeatureMapType featureMapType, const sgl::vk::TexturePtr& featureTexture) {
    if (featureMapType == FeatureMapType::COLOR) {
        inputImageVulkan = featureTexture->getImageView();
    } else {
        sgl::Logfile::get()->writeWarning("Warning in OptixVptDenoiser::setFeatureMap: Unknown feature map.");
    }
}

bool OpenImageDenoiseDenoiser::getUseFeatureMap(FeatureMapType featureMapType) const {
    if (featureMapType == FeatureMapType::COLOR) {
        return true;
    } else {
        return false;
    }
}

void OpenImageDenoiseDenoiser::setUseFeatureMap(FeatureMapType featureMapType, bool useFeature) {
    ;
}

void OpenImageDenoiseDenoiser::setTemporalDenoisingEnabled(bool enabled) {
    ;
}

void OpenImageDenoiseDenoiser::resetFrameNumber() {
    ;
}

void OpenImageDenoiseDenoiser::denoise() {
    if (recreateDenoiserNextFrame) {
        _createDenoiser();
        recreateSwapchain(
                inputImageVulkan->getImage()->getImageSettings().width,
                inputImageVulkan->getImage()->getImageSettings().height);
        recreateDenoiserNextFrame = false;
    }

    renderer->transitionImageLayout(inputImageVulkan, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    inputImageVulkan->getImage()->copyToBuffer(inputImageBufferVk, renderer->getVkCommandBuffer());
    if (supportsMemoryImport) {
        // Sharing semaphores not yet supported. Use device sync.
        renderer->syncWithCpu();
    } else {
        renderer->syncWithCpu();
        auto* inputImageBufferVkPtr = reinterpret_cast<float*>(inputImageBufferVk->mapMemory());
        auto* colorPtr = reinterpret_cast<float*>(oidnGetBufferData(oidnInputColorBuffer));
        memcpy(colorPtr, inputImageBufferVkPtr, inputWidth * inputHeight * 4 * sizeof(float));
        inputImageBufferVk->unmapMemory();
    }


    oidnExecuteFilter(oidnFilter);
    const char* errorMessage = nullptr;
    if (oidnGetDeviceError(oidnDevice, &errorMessage) != OIDN_ERROR_NONE) {
        sgl::Logfile::get()->throwError("Error in OpenImageDenoiseDenoiser::denoise: " + std::string(errorMessage));
    }

    // Sharing semaphores not yet supported. Use oidnSyncDevice.
    oidnSyncDevice(oidnDevice);

    renderer->transitionImageLayout(outputImageVulkan, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    outputImageVulkan->getImage()->copyFromBuffer(outputImageBufferVk, renderer->getVkCommandBuffer());
    if (!supportsMemoryImport) {
        renderer->syncWithCpu();
        auto* outputImageBufferVkPtr = reinterpret_cast<float*>(outputImageBufferVk->mapMemory());
        auto* colorPtr = reinterpret_cast<float*>(oidnGetBufferData(oidnOutputColorBuffer));
        memcpy(outputImageBufferVkPtr, colorPtr, inputWidth * inputHeight * 4 * sizeof(float));
        outputImageBufferVk->unmapMemory();
    }

    // Copy alpha channel, as OpenImageDenoise currently only supports RGB data, or, TODO, separate filter for alpha.
    renderer->transitionImageLayout(inputImageVulkan, VK_IMAGE_LAYOUT_GENERAL);
    renderer->transitionImageLayout(outputImageVulkan, VK_IMAGE_LAYOUT_GENERAL);
    alphaBlitPass->render();
}

void OpenImageDenoiseDenoiser::recreateSwapchain(uint32_t width, uint32_t height) {
    _freeBuffers();
    inputWidth = width;
    inputHeight = height;
    _createBuffers();
    alphaBlitPass->setInputImage(inputImageVulkan);
    alphaBlitPass->setOutputImage(outputImageVulkan);
}


bool OpenImageDenoiseDenoiser::renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) {
    bool reRender = false;

    if (propertyEditor.addCombo(
            "Denoiser Device", (int*)&deviceType, OIDN_DEVICE_TYPE_NAMES, IM_ARRAYSIZE(OIDN_DEVICE_TYPE_NAMES))) {
        reRender = true;
        recreateDenoiserNextFrame = true;
    }

    if (reRender) {
        resetFrameNumber();
    }

    return reRender;
}

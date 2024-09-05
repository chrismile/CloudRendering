/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser, Timm Knörle
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

#include <Utils/AppSettings.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>
#include <Graphics/Vulkan/Render/Passes/Pass.hpp>
#include <Graphics/Vulkan/Render/Passes/BlitComputePass.hpp>
#include <ImGui/imgui.h>
#include <ImGui/Widgets/MultiVarTransferFunctionWindow.hpp>

#include "CloudData.hpp"

#include "PathTracer/VolumetricPathTracingPass.hpp"
#include "PathTracer/LightEditorWidget.hpp"
#include "PathTracer/OccupationVolumePass.hpp"

#include "Config.hpp"
#include "VolumetricPathTracingModuleRenderer.hpp"

#ifdef SUPPORT_CUDA_INTEROP
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#if CUDA_VERSION >= 11020
#define USE_TIMELINE_SEMAPHORES
#elif defined(_WIN32)
#error Binary semaphore sharing is broken on Windows. Please install CUDA >= 11.2 for timeline semaphore support.
#endif
#endif

#ifdef SUPPORT_HIP_INTEROP
#if __has_include(<c10/hip/HIPStream.h>)
#define PYTORCH_HIP_AVAILABLE
#include <c10/hip/HIPStream.h>
#endif
#endif

#ifdef SUPPORT_OPTIX
#include "Denoiser/OptixVptDenoiser.hpp"
#endif
#if defined(SUPPORT_CUDA_INTEROP) && defined(SUPPORT_OPEN_IMAGE_DENOISE)
#include "Denoiser/OpenImageDenoiseDenoiser.hpp"
#endif

ConvertTransmittanceVolumePass::ConvertTransmittanceVolumePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
}

void ConvertTransmittanceVolumePass::setInputOutputData(
        const sgl::vk::ImageViewPtr& _inputImage, const sgl::vk::BufferPtr& _outputBuffer) {
    if (inputImage != _inputImage) {
        inputImage = _inputImage;
        if (computeData) {
            computeData->setStaticImageView(inputImage, "transmittanceVolumeImage");
        }
    }
    if (outputBuffer != _outputBuffer) {
        outputBuffer = _outputBuffer;
        if (computeData) {
            computeData->setStaticBuffer(outputBuffer, "TransmittanceVolumeBuffer");
        }
    }
}

void ConvertTransmittanceVolumePass::clearInputOutputData() {
    inputImage = {};
    outputBuffer = {};
    if (computeData) {
        computeData->setStaticImageView(inputImage, "transmittanceVolumeImage");
        computeData->setStaticBuffer(outputBuffer, "TransmittanceVolumeBuffer");
    }
}

void ConvertTransmittanceVolumePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(BLOCK_SIZE_X)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(BLOCK_SIZE_Y)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(BLOCK_SIZE_Z)));
    std::string shaderName = "ConvertTransmittanceVolume.Compute";
    shaderStages = sgl::vk::ShaderManager->getShaderStages({ shaderName }, preprocessorDefines);
}

void ConvertTransmittanceVolumePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticImageView(inputImage, "transmittanceVolumeImage");
    computeData->setStaticBuffer(outputBuffer, "TransmittanceVolumeBuffer");
}

void ConvertTransmittanceVolumePass::_render() {
    auto xs = inputImage->getImage()->getImageSettings().width;
    auto ys = inputImage->getImage()->getImageSettings().height;
    auto zs = inputImage->getImage()->getImageSettings().depth;
    renderer->dispatch(
            computeData, sgl::uiceil(xs, BLOCK_SIZE_X), sgl::uiceil(ys, BLOCK_SIZE_Y), sgl::uiceil(zs, BLOCK_SIZE_Z));
}



VolumetricPathTracingModuleRenderer::VolumetricPathTracingModuleRenderer(sgl::vk::Renderer* renderer)
        : renderer(renderer) {
    camera = std::make_shared<sgl::Camera>();
    camera->setNearClipDistance(1.0f / 512.0f); // 0.001953125f
    camera->setFarClipDistance(80.0f);
    camera->resetOrientation();
    camera->setPosition(glm::vec3(0.0f, 0.0f, 0.8f));
    camera->setFOVy(std::atan(1.0f / 2.0f) * 2.0f);
    camera->resetLookAtLocation();

    transferFunctionWindow = new sgl::MultiVarTransferFunctionWindow;
    transferFunctionWindow->setShowWindow(false);
    transferFunctionWindow->setAttributeNames({"Volume", "Isosurface"});

    lightEditorWidget = new LightEditorWidget(renderer);

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    vptPass = std::make_shared<VolumetricPathTracingPass>(renderer, &camera);
    renderFinishedFence = std::make_shared<sgl::vk::Fence>(device);

    renderImageBlitPass = std::make_shared<sgl::vk::BlitComputePass>(renderer);
    convertTransmittanceVolumePass = std::make_shared<ConvertTransmittanceVolumePass>(renderer);
}

VolumetricPathTracingModuleRenderer::~VolumetricPathTracingModuleRenderer() {
    if (imageData) {
        delete[] imageData;
        imageData = nullptr;
    }
    if (outputImageBufferCu) {
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        cudaStreamSynchronize(stream);
    }
    renderer->getDevice()->waitIdle();

    vptPass = {};

#ifdef SUPPORT_OPTIX
    if (optixInitialized) {
        OptixVptDenoiser::freeGlobal();
        optixInitialized = false;
    }
#endif
#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::vk::freeCudaDeviceApiFunctionTable();
    }
#endif

    delete transferFunctionWindow;
    transferFunctionWindow = nullptr;
    delete lightEditorWidget;
    lightEditorWidget = nullptr;
    renderer->getDevice()->waitIdle();
}

VolumetricPathTracingPass* VolumetricPathTracingModuleRenderer::getVptPass() {
    return vptPass.get();
}

const CloudDataPtr& VolumetricPathTracingModuleRenderer::getCloudData() {
    return vptPass->getCloudData();
}

void VolumetricPathTracingModuleRenderer::setCloudData(const CloudDataPtr& cloudData) {
    if (storeNextBounds) {
        storeNextBounds = false;
        if (vptPass->getUseSparseGrid()) {
            seq_bounds_min = cloudData->getWorldSpaceSparseGridMin();
            seq_bounds_max = cloudData->getWorldSpaceSparseGridMax();
        } else {
            seq_bounds_min = cloudData->getWorldSpaceDenseGridMin();
            seq_bounds_max = cloudData->getWorldSpaceDenseGridMax();
        }
        hasStoredBounds = true;
    }
    if (hasStoredBounds) {
        cloudData->setSeqBounds(seq_bounds_min, seq_bounds_max);
    }

    vptPass->setCloudData(cloudData);
}

void VolumetricPathTracingModuleRenderer::setEmissionData(const CloudDataPtr& cloudData) {
    if (hasStoredBounds) {
        cloudData->setSeqBounds(seq_bounds_min, seq_bounds_max);
    }
    vptPass->setEmissionData(cloudData);
}

void VolumetricPathTracingModuleRenderer::loadEnvironmentMapImage(const std::string& filename) {
    if (filename.length() == 0) {
        vptPass->setUseEnvironmentMapFlag(false);
    } else {
        vptPass->loadEnvironmentMapImage(filename);
        vptPass->setUseEnvironmentMapFlag(true);
    }
}

void VolumetricPathTracingModuleRenderer::setUseBuiltinEnvironmentMap(const std::string& envMapName) {
    vptPass->setUseEnvironmentMapFlag(false);
    vptPass->setUseBuiltinEnvironmentMap(envMapName);
}

void VolumetricPathTracingModuleRenderer::setEnvironmentMapIntensityFactor(float intensityFactor) {
    vptPass->setEnvironmentMapIntensityFactor(intensityFactor);
}

void VolumetricPathTracingModuleRenderer::setScatteringAlbedo(glm::vec3 albedo) {
    vptPass->setScatteringAlbedo(albedo);
}

void VolumetricPathTracingModuleRenderer::setExtinctionScale(double extinctionScale) {
    vptPass->setExtinctionScale(extinctionScale);
}

void VolumetricPathTracingModuleRenderer::setPhaseG(double phaseG) {
    vptPass->setPhaseG(phaseG);
}

void VolumetricPathTracingModuleRenderer::setExtinctionBase(glm::vec3 extinctionBase) {
    vptPass->setExtinctionBase(extinctionBase);
}

void VolumetricPathTracingModuleRenderer::setUseFeatureMaps(const std::unordered_set<FeatureMapTypeVpt>& featureMapSet) {
    vptPass->setUseFeatureMaps(featureMapSet);
}

void VolumetricPathTracingModuleRenderer::setFeatureMapType(FeatureMapTypeVpt type) {
    vptPass->setFeatureMapType(type);
}

void VolumetricPathTracingModuleRenderer::setDenoiserType(DenoiserType _denoiserType) {
    if (denoiserType != _denoiserType) {
        denoiserType = _denoiserType;
        denoiserSettings.clear();
        isDenoiserDirty = true;
    }
}

void VolumetricPathTracingModuleRenderer::setDenoiserProperty(const std::string& key, const std::string& value) {
    denoiserSettings.insert(std::make_pair(key, value));
    isDenoiserDirty = true;
}

void VolumetricPathTracingModuleRenderer::checkDenoiser() {
    if (isDenoiserDirty) {
        vptPass->setDenoiserType(denoiserType);
        if (!denoiserSettings.empty()) {
            vptPass->setDenoiserSettings(denoiserSettings);
        }
        isDenoiserDirty = false;
    }
}

void VolumetricPathTracingModuleRenderer::setEmissionCap(double emissionCap) {
    vptPass->setEmissionCap(emissionCap);
}

void VolumetricPathTracingModuleRenderer::setEmissionStrength(double emissionStrength) {
    vptPass->setEmissionStrength(emissionStrength);
}

void VolumetricPathTracingModuleRenderer::setUseEmission(bool useEmission) {
    vptPass->setUseEmission(useEmission);
}

void VolumetricPathTracingModuleRenderer::flipYZ(bool flip) {
    vptPass->flipYZ(flip);
}

const glm::vec3& VolumetricPathTracingModuleRenderer::getCameraPosition() {
    return camera->getPosition();
}

const sgl::CameraPtr& VolumetricPathTracingModuleRenderer::getCamera() {
    return camera;
}

void VolumetricPathTracingModuleRenderer::setCameraPosition(const glm::vec3& cameraPosition) {
    this->cameraPosition = cameraPosition;
    camera->setPosition(cameraPosition);
    camera->setLookAtViewMatrix(cameraPosition, cameraTarget, camera->getCameraUp());
    vptPass->onHasMoved();
}

void VolumetricPathTracingModuleRenderer::setCameraTarget(const glm::vec3& cameraTarget) {
    this->cameraTarget = cameraTarget;
    camera->setLookAtViewMatrix(cameraPosition, cameraTarget, camera->getCameraUp());
    vptPass->onHasMoved();
}

void VolumetricPathTracingModuleRenderer::setCameraFOVy(double FOVy) {
    camera->setFOVy(FOVy);
    vptPass->onHasMoved();
}

void VolumetricPathTracingModuleRenderer::setGlobalWorldBoundingBox(const sgl::AABB3& boundingBox) {
    globalWorldBoundingBox = boundingBox;
    hasGlobalWorldBoundingBox = true;
}

void VolumetricPathTracingModuleRenderer::rememberNextBounds() {
    storeNextBounds = true;
}

void VolumetricPathTracingModuleRenderer::forgetCurrentBounds() {
    hasStoredBounds = false;
}

void VolumetricPathTracingModuleRenderer::setViewProjectionMatrixAsPrevious() {
    previousViewProjectionMatrix = camera->getProjectionMatrix() * camera->getViewMatrix();
}

bool VolumetricPathTracingModuleRenderer::settingsDiffer(
        uint32_t width, uint32_t height, uint32_t channels, c10::Device torchDevice, caffe2::TypeMeta dtype) const {
    return !getHasFrameData() || getNumChannels() != channels
           || getFrameWidth() != width || getFrameHeight() != height
           || getDeviceType() != torchDevice.type() || getDType() != dtype;
}

void VolumetricPathTracingModuleRenderer::setRenderingResolution(
        uint32_t width, uint32_t height, uint32_t channels, c10::Device torchDevice, caffe2::TypeMeta dtype) {
    if (dtype != torch::kFloat32) {
        sgl::Logfile::get()->throwError(
                "Error in VolumetricPathTracingModuleRenderer::setRenderingResolution: "
                "The only data type currently supported is 32-bit float.", false);
    }
    this->numChannels = channels;
    this->dtype = dtype;
    this->deviceType = torchDevice.type();
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    if (imageData) {
        delete[] imageData;
        imageData = nullptr;
    }
    if (renderImageView) {
        device->waitIdle();
        renderer->getFrameCommandBuffers();
    }
    renderImageView = {};
    renderImageStaging = {};
    outputImageBufferVk = {};
    outputImageBufferCu = {};
    commandBuffers = {};
    renderReadySemaphore = {};
    renderFinishedSemaphore = {};
    interFrameSemaphores = {};
    timelineValue = 0;

    sgl::vk::ImageSettings imageSettings;
    imageSettings.width = width;
    imageSettings.height = height;
    imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageSettings.usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
            | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    renderImageView = std::make_shared<sgl::vk::ImageView>(std::make_shared<sgl::vk::Image>(
            device, imageSettings));

    //imageSettings.format = numChannels == 3 ? VK_FORMAT_R32G32B32_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;


    // TODO: Add support for not going the GPU->CPU->GPU route with torch::DeviceType::Vulkan.
    if (torchDevice.type() == torch::DeviceType::CPU || torchDevice.type() == torch::DeviceType::Vulkan) {
        imageSettings.tiling = VK_IMAGE_TILING_LINEAR;
        imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageSettings.memoryUsage = VMA_MEMORY_USAGE_GPU_TO_CPU;
        renderImageStaging = std::make_shared<sgl::vk::Image>(device, imageSettings);

        imageData = new float[width * height * numChannels];
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (torchDevice.type() == torch::DeviceType::CUDA) {
        if (!sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
            bool success = sgl::vk::initializeCudaDeviceApiFunctionTable();
            if (!success) {
                sgl::Logfile::get()->throwError(
                        "Error in VolumetricPathTracingModuleRenderer::setRenderingResolution: "
                        "sgl::vk::initializeCudaDeviceApiFunctionTable() failed.", false);
            }
#if defined(SUPPORT_OPTIX) || (defined(SUPPORT_CUDA_INTEROP) && defined(SUPPORT_OPEN_IMAGE_DENOISE))
            if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY
                    && sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
                CUcontext cuContext = {};
                CUdevice cuDevice = {};
                sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuCtxGetCurrent(
                        &cuContext), "Error in cuCtxGetCurrent: ");
                sgl::vk::checkCUresult(sgl::vk::g_cudaDeviceApiFunctionTable.cuCtxGetDevice(
                        &cuDevice), "Error in cuCtxGetDevice: ");
#ifdef SUPPORT_OPTIX
                optixInitialized = OptixVptDenoiser::initGlobal(cuContext, cuDevice);
#endif
#if defined(SUPPORT_CUDA_INTEROP) && defined(SUPPORT_OPEN_IMAGE_DENOISE)
                OpenImageDenoiseDenoiser::initGlobalCuda(cuContext, cuDevice);
#endif
            }
#endif
        }

        outputImageBufferVk = std::make_shared<sgl::vk::Buffer>(
                device, width * height * 4 * sizeof(float),
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                true, true);
        outputImageBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(outputImageBufferVk);
    }
#endif
    else {
        sgl::Logfile::get()->throwError(
                "Error in VolumetricPathTracingModuleRenderer::setRenderingResolution: "
                "Unsupported PyTorch device type.", false);
    }

    vptPass->setOutputImage(renderImageView);
    vptPass->recreateSwapchain(width, height);
    camera->onResolutionChanged(width, height);
}

void VolumetricPathTracingModuleRenderer::createCommandStructures(uint32_t numFrames) {
    //std::cout << "Creating new command structures "<<numFrames<<std::endl;

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    if (renderImageView) {
        device->waitIdle();
        renderer->getFrameCommandBuffers();
    }

    commandBuffers = {};
    renderReadySemaphore = {};
    renderFinishedSemaphore = {};
    interFrameSemaphores = {};
    timelineValue = 0;

    // Use swapchain-like structure where we have N images in flight.
    uint32_t numImages = std::min(numFrames, maxNumFramesInFlight);
    commandBuffers.clear();
    frameFences.clear();
    sgl::vk::CommandPoolType commandPoolType;
    commandPoolType.queueFamilyIndex = device->getComputeQueueIndex();
    commandPoolType.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
#ifdef USE_TIMELINE_SEMAPHORES
    renderReadySemaphore = std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(
            device, 0, VK_SEMAPHORE_TYPE_TIMELINE,
            timelineValue);
    renderFinishedSemaphore = std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(
            device, 0, VK_SEMAPHORE_TYPE_TIMELINE,
            timelineValue);
#else
    renderReadySemaphore = std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(device);
    denoiseFinishedSemaphore = std::make_shared<sgl::vk::SemaphoreVkCudaDriverApiInterop>(device);
#endif
    for (uint32_t frameIdx = 0; frameIdx < numImages; frameIdx++) {
        commandBuffers.push_back(std::make_shared<sgl::vk::CommandBuffer>(device, commandPoolType));
    }
    for (uint32_t frameIdx = 0; frameIdx < numImages; frameIdx++) {
#ifdef USE_TIMELINE_SEMAPHORES
        interFrameSemaphores.push_back(std::make_shared<sgl::vk::Semaphore>(
                device, 0, VK_SEMAPHORE_TYPE_TIMELINE,
                timelineValue));
#else
        interFrameSemaphores.push_back(std::make_shared<sgl::vk::Semaphore>(device));
#endif
        frameFences.push_back(std::make_shared<sgl::vk::Fence>(device, VK_FENCE_CREATE_SIGNALED_BIT));
    }

    //std::cout << "Done creating new command structures "<<numFrames<<std::endl;

}

void VolumetricPathTracingModuleRenderer::setUseSparseGrid(bool useSparseGrid) {
    vptPass->setUseSparseGrid(useSparseGrid);
}

void VolumetricPathTracingModuleRenderer::setGridInterpolationType(GridInterpolationType type) {
    vptPass->setSparseGridInterpolationType(type);
}

void VolumetricPathTracingModuleRenderer::setCustomSeedOffset(uint32_t offset) {
    vptPass->setCustomSeedOffset(offset);
}


void VolumetricPathTracingModuleRenderer::setUseLinearRGB(bool useLinearRGB) {
    vptPass->setUseLinearRGB(useLinearRGB);
}

void VolumetricPathTracingModuleRenderer::setVptMode(VptMode vptMode) {
    vptPass->setVptMode(vptMode);
}

void VolumetricPathTracingModuleRenderer::setVptModeFromString(const std::string& vptModeName) {
    for (int i = 0; i < IM_ARRAYSIZE(VPT_MODE_NAMES); i++) {
        if (vptModeName == VPT_MODE_NAMES[i]) {
            vptPass->setVptMode(VptMode(i));
            return;
        }
    }
    sgl::Logfile::get()->throwError(
            "Error in VolumetricPathTracingModuleRenderer::setVptModeFromString: Unknown VPT mode \""
            + vptModeName + "\".");
}

float* VolumetricPathTracingModuleRenderer::renderFrameCpu(uint32_t numFrames) {
    // TODO: Allow multiple frames in flight.
    for (uint32_t i = 0; i < numFrames; i++) {
        vptPass->setPreviousViewProjMatrix(previousViewProjectionMatrix);
        renderer->beginCommandBuffer();
        vptPass->setIsIntermediatePass(i != numFrames - 1);
        // Needs to be called here as we re-render in the same frame a new denoiser was set.
        vptPass->checkRecreateDenoiser();
        vptPass->render();
        if (i == numFrames - 1) {
            renderer->transitionImageLayout(renderImageView, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            renderer->transitionImageLayout(renderImageStaging, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            renderImageView->getImage()->copyToImage(
                    renderImageStaging, VK_IMAGE_ASPECT_COLOR_BIT,
                    renderer->getVkCommandBuffer());
        }
        renderer->endCommandBuffer();

        // Submit the rendering operations in Vulkan.
        renderer->submitToQueue(
                {}, {}, renderFinishedFence,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        renderFinishedFence->wait();
        renderFinishedFence->reset();
    }

    uint32_t width = renderImageStaging->getImageSettings().width;
    uint32_t height = renderImageStaging->getImageSettings().height;
    VkSubresourceLayout subresourceLayout =
            renderImageStaging->getSubresourceLayout(VK_IMAGE_ASPECT_COLOR_BIT);
    auto* mappedData = (float*)renderImageStaging->mapMemory();
    const auto rowPitch = uint32_t(subresourceLayout.rowPitch / sizeof(float));

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t writeLocation = (x + y * width) * numChannels;
            // We don't need to add "uint32_t(subresourceLayout.offset)" here, as this is automatically done by VMA.
            uint32_t readLocation = x * 4 + rowPitch * y;
            for (uint32_t c = 0; c < numChannels; c++) {
                imageData[writeLocation + c] = mappedData[readLocation + c];
            }
        }
    }

    renderImageStaging->unmapMemory();
    return imageData;
}

float* VolumetricPathTracingModuleRenderer::renderFrameVulkan(uint32_t numFrames) {
    // TODO: Add support for not going the GPU->CPU->GPU route.
    return nullptr;
}

#ifdef SUPPORT_CUDA_INTEROP
float* VolumetricPathTracingModuleRenderer::renderFrameCuda(uint32_t numFrames) {
    if (!sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        sgl::Logfile::get()->throwError(
                "Error in VolumetricPathTracingModuleRenderer::renderFrameCuda: "
                "sgl::vk::getIsCudaDeviceApiFunctionTableInitialized() returned false.", false);
    }

    uint32_t numImages = std::min(numFrames, maxNumFramesInFlight);
    if (size_t(numImages) > commandBuffers.size()) {
        this->createCommandStructures(numImages);

        //sgl::Logfile::get()->throwError(
        //        "Error in VolumetricPathTracingModuleRenderer::renderFrameCuda: Frame data was not allocated.",
        //        false);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    timelineValue++;
    uint64_t waitValue = timelineValue;

#ifdef USE_TIMELINE_SEMAPHORES
    renderReadySemaphore->signalSemaphoreCuda(stream, timelineValue);
#else
    renderReadySemaphores->signalSemaphoreCuda(stream);
#endif

    for (uint32_t frameIndex = 0; frameIndex < numFrames; frameIndex++) {
        const uint32_t imageIndex = frameIndex % maxNumFramesInFlight;
        auto& fence = frameFences.at(imageIndex);
        if (frameIndex == maxNumFramesInFlight) {
            sgl::vk::g_cudaDeviceApiFunctionTable.cuStreamSynchronize(stream);
        }
        if (frameIndex != 0 && imageIndex == 0) {
            waitValue = timelineValue;
            timelineValue++;
        }
        fence->wait();
        fence->reset();

        sgl::vk::CommandBufferPtr commandBuffer = commandBuffers.at(imageIndex);
        sgl::vk::SemaphorePtr waitSemaphore;
        sgl::vk::SemaphorePtr signalSemaphore;
        VkPipelineStageFlags waitStage;
        if (frameIndex == 0) {
            waitSemaphore = renderReadySemaphore;
            waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        } else {
            waitSemaphore = interFrameSemaphores.at((imageIndex + maxNumFramesInFlight - 1) % maxNumFramesInFlight);
            waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
        if (frameIndex == numFrames - 1) {
            signalSemaphore = renderFinishedSemaphore;
        } else {
            signalSemaphore = interFrameSemaphores.at(imageIndex);
        }
#ifdef USE_TIMELINE_SEMAPHORES
        waitSemaphore->setWaitSemaphoreValue(waitValue);
        signalSemaphore->setSignalSemaphoreValue(timelineValue);
#endif
        commandBuffer->pushWaitSemaphore(waitSemaphore, waitStage);
        commandBuffer->pushSignalSemaphore(signalSemaphore);
        commandBuffer->setFence(fence);

        renderer->pushCommandBuffer(commandBuffer);
        renderer->beginCommandBuffer();

        vptPass->setPreviousViewProjMatrix(previousViewProjectionMatrix);

        vptPass->setIsIntermediatePass(frameIndex != numFrames - 1);
        // Needs to be called here as we re-render in the same frame a new denoiser was set.
        vptPass->checkRecreateDenoiser();
        vptPass->render();

        if (frameIndex == numFrames - 1) {
            renderer->transitionImageLayout(renderImageView, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            renderImageView->getImage()->copyToBuffer(
                    outputImageBufferVk, renderer->getVkCommandBuffer());
        }

        renderer->endCommandBuffer();

        renderer->submitToQueue();
    }

#ifdef USE_TIMELINE_SEMAPHORES
    renderFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);
#else
    renderFinishedSemaphore->waitSemaphoreCuda(stream);
#endif

    // TODO
    renderer->getDevice()->waitIdle();
    cudaStreamSynchronize(stream);

    return (float*)outputImageBufferCu->getCudaDevicePtr();
}
#endif

float* VolumetricPathTracingModuleRenderer::getFeatureMapCpu(FeatureMapTypeVpt featureMap) {
    sgl::vk::TexturePtr texture = vptPass->getFeatureMapTexture(featureMap);
    
    renderer->beginCommandBuffer();

    if (featureMap == FeatureMapTypeVpt::TRANSMITTANCE_VOLUME) {
        sgl::Logfile::get()->throwError(
                "Error in VolumetricPathTracingModuleRenderer::getFeatureMapCpu: Transmittance volume "
                "is currently only supported with CUDA.");
    } else {
        if (renderer->getUseGraphicsQueue()) {
            renderer->transitionImageLayout(texture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            renderer->transitionImageLayout(renderImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            texture->getImage()->blit(renderImageView->getImage(), renderer->getVkCommandBuffer());
        } else {
            renderer->transitionImageLayout(texture->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            renderer->transitionImageLayout(renderImageView->getImage(), VK_IMAGE_LAYOUT_GENERAL);
            // The commands below only write to the descriptor set when the texture/image changes.
            // In this case, waitIdle makes sure the descriptor set is not currently in use.
            renderImageBlitPass->setInputTexture(texture);
            renderImageBlitPass->setOutputImage(renderImageView);
            renderImageBlitPass->render();
        }

        renderer->transitionImageLayout(renderImageView, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(renderImageStaging, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        renderImageView->getImage()->copyToImage(
                renderImageStaging, VK_IMAGE_ASPECT_COLOR_BIT,
                renderer->getVkCommandBuffer());
    }

    renderer->endCommandBuffer();

    // Submit the rendering operations in Vulkan.
    renderer->submitToQueue(
            {}, {}, renderFinishedFence,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    renderFinishedFence->wait();
    renderFinishedFence->reset();
    
    uint32_t width = renderImageStaging->getImageSettings().width;
    uint32_t height = renderImageStaging->getImageSettings().height;
    VkSubresourceLayout subresourceLayout =
            renderImageStaging->getSubresourceLayout(VK_IMAGE_ASPECT_COLOR_BIT);
    auto* mappedData = (float*)renderImageStaging->mapMemory();
    const auto rowPitch = uint32_t(subresourceLayout.rowPitch / sizeof(float));

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t writeLocation = (x + y * width) * numChannels;
            // We don't need to add "uint32_t(subresourceLayout.offset)" here, as this is automatically done by VMA.
            uint32_t readLocation = x * 4 + rowPitch * y;
            for (uint32_t c = 0; c < numChannels; c++) {
                imageData[writeLocation + c] = mappedData[readLocation + c];
            }
        }
    }

    renderImageStaging->unmapMemory();
    return imageData;
}


float* VolumetricPathTracingModuleRenderer::getFeatureMapCuda(FeatureMapTypeVpt featureMap) {
    sgl::vk::TexturePtr texture = vptPass->getFeatureMapTexture(featureMap);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // TODO
    renderer->getDevice()->waitIdle();
    cudaStreamSynchronize(stream);

    timelineValue++;

    sgl::vk::CommandBufferPtr commandBuffer = commandBuffers.at(0);
    sgl::vk::SemaphorePtr waitSemaphore;
    sgl::vk::SemaphorePtr signalSemaphore;

    VkPipelineStageFlags waitStage;
    waitSemaphore = renderReadySemaphore;
    waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    signalSemaphore = renderFinishedSemaphore;
    
#ifdef USE_TIMELINE_SEMAPHORES
    waitSemaphore->setWaitSemaphoreValue(timelineValue);
    signalSemaphore->setSignalSemaphoreValue(timelineValue);
#endif
    commandBuffer->pushWaitSemaphore(waitSemaphore, waitStage);
    commandBuffer->pushSignalSemaphore(signalSemaphore);

    renderer->pushCommandBuffer(commandBuffer);
    renderer->beginCommandBuffer();

    if (featureMap == FeatureMapTypeVpt::TRANSMITTANCE_VOLUME) {
        bool recreate = !outputVolumeBufferVk;
        size_t secondaryVolumeSizeInBytes = vptPass->getSecondaryVolumeSizeInBytes();
        if (!recreate) {
            if (secondaryVolumeSizeInBytes != outputVolumeBufferVk->getSizeInBytes()) {
                recreate = true;
            }
        }
        if (recreate) {
            outputVolumeBufferCu = {};
            outputVolumeBufferVk = {};
            convertTransmittanceVolumePass->clearInputOutputData();
            outputVolumeBufferVk = std::make_shared<sgl::vk::Buffer>(
                    renderer->getDevice(), secondaryVolumeSizeInBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                    true, true);
            outputVolumeBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(outputVolumeBufferVk);
        }
        convertTransmittanceVolumePass->setInputOutputData(texture->getImageView(), outputVolumeBufferVk);
        renderer->insertImageMemoryBarrier(
                texture->getImage(),
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        convertTransmittanceVolumePass->render();
    } else {
        if (renderer->getUseGraphicsQueue()) {
            renderer->transitionImageLayout(texture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            renderer->transitionImageLayout(renderImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            texture->getImage()->blit(renderImageView->getImage(), renderer->getVkCommandBuffer());
        } else {
            renderer->transitionImageLayout(texture->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            renderer->transitionImageLayout(renderImageView->getImage(), VK_IMAGE_LAYOUT_GENERAL);
            // The commands below only write to the descriptor set when the texture/image changes.
            // In this case, waitIdle makes sure the descriptor set is not currently in use.
            renderImageBlitPass->setInputTexture(texture);
            renderImageBlitPass->setOutputImage(renderImageView);
            renderImageBlitPass->render();
        }

        renderer->transitionImageLayout(renderImageView, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderImageView->getImage()->copyToBuffer(
                outputImageBufferVk, renderer->getVkCommandBuffer());
    }

    renderer->endCommandBuffer();
    renderer->submitToQueue();

#ifdef USE_TIMELINE_SEMAPHORES
    renderReadySemaphore->signalSemaphoreCuda(stream, timelineValue);
#else
    renderReadySemaphores->signalSemaphoreCuda(stream);
#endif

#ifdef USE_TIMELINE_SEMAPHORES
    renderFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);
#else
    renderFinishedSemaphore->waitSemaphoreCuda(stream);
#endif

    // TODO
    cudaStreamSynchronize(stream);
    renderer->getDevice()->waitIdle();

    if (featureMap == FeatureMapTypeVpt::TRANSMITTANCE_VOLUME) {
        return (float*)outputVolumeBufferCu->getCudaDevicePtr();
    } else {
        return (float*)outputImageBufferCu->getCudaDevicePtr();
    }
}

float* VolumetricPathTracingModuleRenderer::computeOccupationVolumeCuda(uint32_t ds, uint32_t maxKernelRadius) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // TODO
    renderer->getDevice()->waitIdle();
    cudaStreamSynchronize(stream);

    timelineValue++;

    sgl::vk::CommandBufferPtr commandBuffer = commandBuffers.at(0);
    sgl::vk::SemaphorePtr waitSemaphore;
    sgl::vk::SemaphorePtr signalSemaphore;

    VkPipelineStageFlags waitStage;
    waitSemaphore = renderReadySemaphore;
    waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    signalSemaphore = renderFinishedSemaphore;

#ifdef USE_TIMELINE_SEMAPHORES
    waitSemaphore->setWaitSemaphoreValue(timelineValue);
    signalSemaphore->setSignalSemaphoreValue(timelineValue);
#endif
    commandBuffer->pushWaitSemaphore(waitSemaphore, waitStage);
    commandBuffer->pushSignalSemaphore(signalSemaphore);

    renderer->pushCommandBuffer(commandBuffer);
    renderer->beginCommandBuffer();

    const auto& cloudData = vptPass->getCloudData();
    size_t occupationVolumeSizeInBytes =
            sgl::uiceil(cloudData->getGridSizeX(), ds) * sgl::uiceil(cloudData->getGridSizeY(), ds) *
            sgl::uiceil(cloudData->getGridSizeZ(), ds) * sizeof(uint8_t);
    bool recreate = !outputOccupationVolumeBufferVk;
    if (!recreate) {
        if (occupationVolumeSizeInBytes != outputOccupationVolumeBufferVk->getSizeInBytes()) {
            recreate = true;
        }
    }
    if (recreate) {
        outputOccupationVolumeBufferCu = {};
        outputOccupationVolumeBufferVk = {};
        convertTransmittanceVolumePass->clearInputOutputData();
        outputOccupationVolumeBufferVk = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), occupationVolumeSizeInBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_ONLY,
                true, true);
        outputOccupationVolumeBufferCu = std::make_shared<sgl::vk::BufferCudaDriverApiExternalMemoryVk>(outputOccupationVolumeBufferVk);
    }

    auto* occupationVolumePass = new OccupationVolumePass(renderer);
    auto occupationVolumeImage = occupationVolumePass->computeVolume(vptPass.get(), ds, maxKernelRadius);

    occupationVolumeImage->getImage()->copyToBuffer(
            outputOccupationVolumeBufferVk, renderer->getVkCommandBuffer());

    renderer->endCommandBuffer();
    renderer->submitToQueue();

#ifdef USE_TIMELINE_SEMAPHORES
    renderReadySemaphore->signalSemaphoreCuda(stream, timelineValue);
#else
    renderReadySemaphores->signalSemaphoreCuda(stream);
#endif

#ifdef USE_TIMELINE_SEMAPHORES
    renderFinishedSemaphore->waitSemaphoreCuda(stream, timelineValue);
#else
    renderFinishedSemaphore->waitSemaphoreCuda(stream);
#endif

    // TODO
    cudaStreamSynchronize(stream);
    renderer->getDevice()->waitIdle();

    delete occupationVolumePass;

    return (float*)outputOccupationVolumeBufferCu->getCudaDevicePtr();
}

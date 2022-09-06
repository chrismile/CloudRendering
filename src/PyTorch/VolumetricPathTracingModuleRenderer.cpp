/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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
#include <ImGui/imgui.h>

#include "PathTracer/VolumetricPathTracingPass.hpp"

#include "Config.hpp"
#include "VolumetricPathTracingModuleRenderer.hpp"

#ifdef SUPPORT_CUDA_INTEROP
#include <cuda.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDAStream.h>
#if CUDA_VERSION >= 11020
#define USE_TIMELINE_SEMAPHORES
#elif defined(_WIN32)
#error Binary semaphore sharing is broken on Windows. Please install CUDA >= 11.2 for timeline semaphore support.
#endif
#endif

VolumetricPathTracingModuleRenderer::VolumetricPathTracingModuleRenderer(sgl::vk::Renderer* renderer)
        : renderer(renderer) {
    camera = std::make_shared<sgl::Camera>();
    camera->setNearClipDistance(0.01f);
    camera->setFarClipDistance(100.0f);
    camera->setOrientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
    camera->setYaw(-sgl::PI / 2.0f); //< around y axis
    camera->setPitch(0.0f); //< around x axis
    camera->setPosition(glm::vec3(0.0f, 0.0f, 0.8f));
    camera->setFOVy(std::atan(1.0f / 2.0f) * 2.0f);
    camera->resetLookAtLocation();

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    vptPass = std::make_shared<VolumetricPathTracingPass>(renderer, &camera);
    renderFinishedFence = std::make_shared<sgl::vk::Fence>(device);
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
}

void VolumetricPathTracingModuleRenderer::setCloudData(const CloudDataPtr& cloudData) {
    vptPass->setCloudData(cloudData);
}

void VolumetricPathTracingModuleRenderer::loadEnvironmentMapImage(const std::string& filename) {
    if (filename.length() == 0){
        vptPass->setUseEnvironmentMapFlag(false);
    } else {
        vptPass->loadEnvironmentMapImage(filename);
        vptPass->setUseEnvironmentMapFlag(true);
    }
}

void VolumetricPathTracingModuleRenderer::setEnvironmentMapIntensityFactor(float intensityFactor){
    vptPass->setEnvironmentMapIntensityFactor(intensityFactor);
}

void VolumetricPathTracingModuleRenderer::setScatteringAlbedo(glm::vec3 albedo){
    vptPass->setScatteringAlbedo(albedo);
}

void VolumetricPathTracingModuleRenderer::setExtinctionScale(double extinctionScale){
    vptPass->setExtinctionScale(extinctionScale);
}

void VolumetricPathTracingModuleRenderer::setPhaseG(double phaseG){
    vptPass->setPhaseG(phaseG);
}

void VolumetricPathTracingModuleRenderer::setExtinctionBase(glm::vec3 extinctionBase){
    vptPass->setExtinctionBase(extinctionBase);
}

void VolumetricPathTracingModuleRenderer::setFeatureMapType(FeatureMapTypeVpt type){
    vptPass->setFeatureMapType(type);
}


void VolumetricPathTracingModuleRenderer::setCameraPosition(glm::vec3 cameraPosition){
    this->cameraPosition = cameraPosition;
    camera->setPosition(cameraPosition);
    camera->setLookAtViewMatrix(cameraPosition, cameraTarget, camera->getCameraUp());
}

void VolumetricPathTracingModuleRenderer::setCameraTarget(glm::vec3 cameraTarget){
    this->cameraTarget = cameraTarget;
    camera->setLookAtViewMatrix(cameraPosition, cameraTarget, camera->getCameraUp());
}

void VolumetricPathTracingModuleRenderer::setCameraFOVy(double FOVy) {
    camera->setFOVy(FOVy);
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

    // TODO: Use swapchain-like structure where we have N images in flight.
    size_t numImages = numFrames;
    commandBuffers.clear();
    sgl::vk::CommandPoolType commandPoolType;
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
    for (size_t frameIdx = 0; frameIdx < numImages; frameIdx++) {
        commandBuffers.push_back(std::make_shared<sgl::vk::CommandBuffer>(device, commandPoolType));
    }
    for (size_t frameIdx = 0; frameIdx < numImages - 1; frameIdx++) {
#ifdef USE_TIMELINE_SEMAPHORES
        interFrameSemaphores.push_back(std::make_shared<sgl::vk::Semaphore>(
                device, 0, VK_SEMAPHORE_TYPE_TIMELINE,
                timelineValue));
#else
        interFrameSemaphores.push_back(std::make_shared<sgl::vk::Semaphore>(device));
#endif
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
        vptPass->render();
        if (i == numFrames - 1) {
            renderImageView->getImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, renderer->getVkCommandBuffer());
            renderImageStaging->transitionImageLayout(
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, renderer->getVkCommandBuffer());
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

    if (size_t(numFrames) > commandBuffers.size() || size_t(numFrames) > interFrameSemaphores.size() + 1) {
        this->createCommandStructures(numFrames);

        //sgl::Logfile::get()->throwError(
        //        "Error in VolumetricPathTracingModuleRenderer::renderFrameCuda: Frame data was not allocated.",
        //        false);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    timelineValue++;

    for (uint32_t frameIndex = 0; frameIndex < numFrames; frameIndex++) {
        sgl::vk::CommandBufferPtr commandBuffer = commandBuffers.at(frameIndex);
        sgl::vk::SemaphorePtr waitSemaphore;
        sgl::vk::SemaphorePtr signalSemaphore;
        VkPipelineStageFlags waitStage;
        if (frameIndex == 0) {
            waitSemaphore = renderReadySemaphore;
            waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        } else {
            waitSemaphore = interFrameSemaphores.at(frameIndex - 1);
            waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        }
        if (frameIndex == numFrames - 1) {
            signalSemaphore = renderFinishedSemaphore;
        } else {
            signalSemaphore = interFrameSemaphores.at(frameIndex);
        }
#ifdef USE_TIMELINE_SEMAPHORES
        waitSemaphore->setWaitSemaphoreValue(timelineValue);
        signalSemaphore->setSignalSemaphoreValue(timelineValue);
#endif
        commandBuffer->pushWaitSemaphore(waitSemaphore, waitStage);
        commandBuffer->pushSignalSemaphore(signalSemaphore);

        renderer->pushCommandBuffer(commandBuffer);
        renderer->beginCommandBuffer();

        vptPass->setPreviousViewProjMatrix(previousViewProjectionMatrix);

        vptPass->render();

        if (frameIndex == numFrames - 1) {
            renderImageView->getImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, renderer->getVkCommandBuffer());
            renderImageView->getImage()->copyToBuffer(
                    outputImageBufferVk, renderer->getVkCommandBuffer());
        }

        renderer->endCommandBuffer();
    }
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

    //renderer->getDevice()->waitIdle();
    //cudaStreamSynchronize(stream);
    return (float*)outputImageBufferCu->getCudaDevicePtr();
}
#endif

float* VolumetricPathTracingModuleRenderer::getFeatureMapCpu(FeatureMapTypeVpt featureMap) {

    sgl::vk::TexturePtr texture = vptPass->getFeatureMapTexture(featureMap);
    
    renderer->beginCommandBuffer();

    renderer->transitionImageLayout(texture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    renderer->transitionImageLayout(renderImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    texture->getImage()->blit(renderImageView->getImage(), renderer->getVkCommandBuffer());

    renderImageView->getImage()->transitionImageLayout(
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, renderer->getVkCommandBuffer());
    renderImageStaging->transitionImageLayout(
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, renderer->getVkCommandBuffer());
    renderImageView->getImage()->copyToImage(
            renderImageStaging, VK_IMAGE_ASPECT_COLOR_BIT,
            renderer->getVkCommandBuffer());
    
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

    renderer->transitionImageLayout(texture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    renderer->transitionImageLayout(renderImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    texture->getImage()->blit(renderImageView->getImage(), renderer->getVkCommandBuffer());

    renderImageView->getImage()->transitionImageLayout(
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, renderer->getVkCommandBuffer());
    renderImageView->getImage()->copyToBuffer(
            outputImageBufferVk, renderer->getVkCommandBuffer());

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

    //renderer->getDevice()->waitIdle();
    //cudaStreamSynchronize(stream);
    return (float*)outputImageBufferCu->getCudaDevicePtr();
    
}
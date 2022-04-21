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
#include <Graphics/Scene/RenderTarget.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Utils/Swapchain.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/imgui_impl_vulkan.h>
#include "PathTracer/VolumetricPathTracingPass.hpp"
#include "DataView.hpp"

DataView::DataView(
        sgl::CameraPtr& parentCamera, sgl::vk::Renderer* rendererVk,
        const std::shared_ptr<VolumetricPathTracingPass>& volumetricPathTracingPass)
        : parentCamera(parentCamera), rendererVk(rendererVk), volumetricPathTracingPass(volumetricPathTracingPass) {
    device = rendererVk->getDevice();

    camera = std::make_shared<sgl::Camera>();
    camera->copyState(parentCamera);

    sceneTextureBlitPass = std::make_shared<sgl::vk::BlitRenderPass>(rendererVk);
    sceneTextureGammaCorrectionPass = sgl::vk::BlitRenderPassPtr(new sgl::vk::BlitRenderPass(
            rendererVk, {"GammaCorrection.Vertex", "GammaCorrection.Fragment"}));

    screenshotReadbackHelper = std::make_shared<sgl::vk::ScreenshotReadbackHelper>(rendererVk);
}

DataView::~DataView() {
    if (descriptorSetImGui) {
        sgl::ImGuiWrapper::get()->freeDescriptorSet(descriptorSetImGui);
        descriptorSetImGui = nullptr;
    }
}

void DataView::resize(int newWidth, int newHeight) {
    viewportWidth = uint32_t(std::max(newWidth, 0));
    viewportHeight = uint32_t(std::max(newHeight, 0));

    if (viewportWidth == 0 || viewportHeight == 0) {
        dataViewTexture = {};
        compositedDataViewTexture = {};
        return;
    }
    auto renderTarget = std::make_shared<sgl::RenderTarget>(
            int(viewportWidth), int(viewportHeight));
    camera->setRenderTarget(renderTarget, false);
    camera->onResolutionChanged({});

    sgl::vk::ImageSettings imageSettings;
    imageSettings.width = viewportWidth;
    imageSettings.height = viewportHeight;

    // Create scene texture.
    imageSettings.usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    if (useLinearRGB) {
        imageSettings.format = VK_FORMAT_R16G16B16A16_UNORM;
    } else {
        imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
    }
    dataViewTexture = std::make_shared<sgl::vk::Texture>(
            device, imageSettings, sgl::vk::ImageSamplerSettings(),
            VK_IMAGE_ASPECT_COLOR_BIT);
    dataViewTexture->getImage()->transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    dataViewTexture->getImageView()->clearColor(
            clearColor.getFloatColorRGBA(), rendererVk->getVkCommandBuffer());

    // Create composited (gamma-resolved, if VK_FORMAT_R16G16B16A16_UNORM for scene texture) scene texture.
    imageSettings.usage =
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
    compositedDataViewTexture = std::make_shared<sgl::vk::Texture>(
            device, imageSettings, sgl::vk::ImageSamplerSettings(),
            VK_IMAGE_ASPECT_COLOR_BIT);

    // Pass the textures to the render passes.
    sceneTextureBlitPass->setInputTexture(dataViewTexture);
    sceneTextureBlitPass->setOutputImage(compositedDataViewTexture->getImageView());
    sceneTextureBlitPass->recreateSwapchain(viewportWidth, viewportHeight);

    sceneTextureGammaCorrectionPass->setInputTexture(dataViewTexture);
    sceneTextureGammaCorrectionPass->setOutputImage(compositedDataViewTexture->getImageView());
    sceneTextureGammaCorrectionPass->recreateSwapchain(viewportWidth, viewportHeight);

    screenshotReadbackHelper->onSwapchainRecreated(viewportWidth, viewportHeight);

    if (descriptorSetImGui) {
        sgl::ImGuiWrapper::get()->freeDescriptorSet(descriptorSetImGui);
        descriptorSetImGui = nullptr;
    }
    descriptorSetImGui = ImGui_ImplVulkan_AddTexture(
            compositedDataViewTexture->getImageSampler()->getVkSampler(),
            compositedDataViewTexture->getImageView()->getVkImageView(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    volumetricPathTracingPass->setOutputImage(dataViewTexture->getImageView());
    volumetricPathTracingPass->recreateSwapchain(viewportWidth, viewportHeight);
}

void DataView::beginRender() {
    camera->copyState(parentCamera);
}

void DataView::endRender() {
    rendererVk->transitionImageLayout(
            dataViewTexture->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    if (useLinearRGB) {
        sceneTextureGammaCorrectionPass->render();
    } else {
        sceneTextureBlitPass->render();
    }
}

void DataView::syncCamera() {
    camera->copyState(parentCamera);
}

void DataView::saveScreenshot(const std::string& filename) {
    rendererVk->transitionImageLayout(
            compositedDataViewTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    screenshotReadbackHelper->requestScreenshotReadback(compositedDataViewTexture->getImage(), filename);
}

void DataView::saveScreenshotDataIfAvailable() {
    if (viewportWidth == 0 || viewportHeight == 0) {
        return;
    }
    sgl::vk::Swapchain* swapchain = sgl::AppSettings::get()->getSwapchain();
    uint32_t imageIndex = swapchain ? swapchain->getImageIndex() : 0;
    screenshotReadbackHelper->saveDataIfAvailable(imageIndex);
}

ImTextureID DataView::getImGuiTextureId() const {
    compositedDataViewTexture->getImage()->transitionImageLayout(
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, rendererVk->getVkCommandBuffer());
    return reinterpret_cast<ImTextureID>(descriptorSetImGui);
}

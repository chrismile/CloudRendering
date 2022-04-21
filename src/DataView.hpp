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

#ifndef CLOUDRENDERING_DATAVIEW_HPP
#define CLOUDRENDERING_DATAVIEW_HPP

#include <Graphics/Color.hpp>
#include <Graphics/Scene/Camera.hpp>
#include <Graphics/Vulkan/Utils/ScreenshotReadbackHelper.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Shader/Shader.hpp>
#include <Graphics/Vulkan/Render/Passes/BlitRenderPass.hpp>
#include <ImGui/imgui.h>

class VolumetricPathTracingPass;

class DataView {
public:
    DataView(
            sgl::CameraPtr& parentCamera, sgl::vk::Renderer* rendererVk,
            const std::shared_ptr<VolumetricPathTracingPass>& volumetricPathTracingPass);
    ~DataView();
    void resize(int newWidth, int newHeight);
    void beginRender();
    void endRender();
    void syncCamera();
    void saveScreenshot(const std::string& filename);
    void saveScreenshotDataIfAvailable();
    [[nodiscard]] ImTextureID getImGuiTextureId() const;

    sgl::CameraPtr parentCamera;
    sgl::CameraPtr camera;
    bool useLinearRGB = false;
    bool showWindow = true;
    sgl::Color clearColor;
    uint32_t viewportWidth = 0;
    uint32_t viewportHeight = 0;

    sgl::vk::TexturePtr dataViewTexture; ///< Can be 8 or 16 bits per pixel.
    sgl::vk::TexturePtr compositedDataViewTexture; ///< The final RGBA8 texture.
    sgl::vk::BlitRenderPassPtr sceneTextureBlitPass;
    sgl::vk::BlitRenderPassPtr sceneTextureGammaCorrectionPass;

    sgl::vk::ScreenshotReadbackHelperPtr screenshotReadbackHelper; ///< For reading back screenshots from the GPU.

    VkDescriptorSet descriptorSetImGui{};

private:
    sgl::vk::Renderer* rendererVk;
    sgl::vk::Device* device;
    std::shared_ptr<VolumetricPathTracingPass> volumetricPathTracingPass;
};


#endif //CLOUDRENDERING_DATAVIEW_HPP

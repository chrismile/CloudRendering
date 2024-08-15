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

#ifndef CLOUDRENDERING_OPENIMAGEDENOISEDENOISER_HPP
#define CLOUDRENDERING_OPENIMAGEDENOISEDENOISER_HPP

#include <Graphics/Vulkan/Render/Renderer.hpp>

#include <OpenImageDenoise/oidn.h>

#include "Denoiser.hpp"

enum class OIDNDeviceTypeCustom {
    DEFAULT = 0, GPU_UUID = 1, CPU = 2, SYCL = 3, CUDA = 4, HIP = 5
#ifdef __APPLE__
    , METAL = 6
#endif
};

const char* const OIDN_DEVICE_TYPE_NAMES[] = {
    "Default", "GPU by UUID", "CPU", "SYCL", "CUDA", "HIP"
#ifdef __APPLE__
    , "Metal"
#endif
};

namespace sgl { namespace vk {
class BufferCustomInteropVk;
typedef std::shared_ptr<BufferCustomInteropVk> BufferCustomInteropVkPtr;
}}

class AlphaBlitPass;

class OpenImageDenoiseDenoiser : public Denoiser {
public:
    explicit OpenImageDenoiseDenoiser(sgl::vk::Renderer* renderer);
    ~OpenImageDenoiseDenoiser() override;

    [[nodiscard]] DenoiserType getDenoiserType() const override { return DenoiserType::OPEN_IMAGE_DENOISE; }
    [[nodiscard]] const char* getDenoiserName() const override { return "OpenImageDenoise"; }
    void setOutputImage(sgl::vk::ImageViewPtr& outputImage) override;
    void setFeatureMap(FeatureMapType featureMapType, const sgl::vk::TexturePtr& featureTexture) override;
    [[nodiscard]] bool getUseFeatureMap(FeatureMapType featureMapType) const override;
    void setUseFeatureMap(FeatureMapType featureMapType, bool useFeature) override;
    void setTemporalDenoisingEnabled(bool enabled) override; //< Call if renderer doesn't support temporal denoising.
    void resetFrameNumber() override;
    void denoise() override;
    void recreateSwapchain(uint32_t width, uint32_t height) override;

    /// Renders the GUI. Returns whether re-rendering has become necessary due to the user's actions.
    bool renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) override;

private:
    sgl::vk::Renderer* renderer = nullptr;
    std::shared_ptr<AlphaBlitPass> alphaBlitPass;
    sgl::vk::ImageViewPtr inputImageVulkan, outputImageVulkan;
    sgl::vk::BufferPtr inputImageBufferVk, outputImageBufferVk;
    sgl::vk::BufferCustomInteropVkPtr inputImageBufferInterop, outputImageBufferInterop;

    // OpenImageDenoise data.
    void _createDenoiser();
    void _freeDenoiser();
    void _createBuffers();
    void _freeBuffers();
    uint32_t inputWidth = 0;
    uint32_t inputHeight = 0;
    OIDNDeviceTypeCustom deviceType = OIDNDeviceTypeCustom::GPU_UUID;
    bool supportsMemoryImport = false;
    bool recreateDenoiserNextFrame = false;
    OIDNDevice oidnDevice{};
    OIDNBuffer oidnInputColorBuffer{};
    OIDNBuffer oidnOutputColorBuffer{};
    OIDNFilter oidnFilter{};
};

#endif //CLOUDRENDERING_OPENIMAGEDENOISEDENOISER_HPP

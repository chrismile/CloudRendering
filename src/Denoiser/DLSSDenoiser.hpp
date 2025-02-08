/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2025, Christoph Neuhauser
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

#ifndef CLOUDRENDERING_DLSSDENOISER_HPP
#define CLOUDRENDERING_DLSSDENOISER_HPP

#include <Graphics/Vulkan/Render/Renderer.hpp>

#include "Denoiser.hpp"

struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Parameter;

namespace sgl { namespace vk {
class Device;
class ImageView;
typedef std::shared_ptr<ImageView> ImageViewPtr;
}}

enum class DlssPerfQuality {
    MAX_PERF, BALANCED, MAX_QUALITY, ULTRA_PERFORMANCE, ULTRA_QUALITY, DLAA
};
enum class DlssRenderPreset {
    DEFAULT, // all
    A, // Deprected
    B, // Deprected
    C, // Deprected
    D, // Transformer model (since SDK version 310.1.0)
};
const char* const DLSS_MODEL_TYPE_NAMES[] = {
    "CNN", "Transformer"
};

sgl::vk::PhysicalDeviceCheckCallback getDlssPhysicalDeviceCheckCallback(sgl::vk::Instance* instance);
bool getIsDlssSupportedByDevice(sgl::vk::Device* device);

/*
 * TODO: Jittering may need to be specified, also for motion vectors.
 * Only DLAA for now, as render res == output res?
 */

class DLSSDenoiser : public Denoiser {
public:
    explicit DLSSDenoiser(sgl::vk::Renderer* renderer, bool _denoiseAlpha);
    ~DLSSDenoiser() override;

    [[nodiscard]] DenoiserType getDenoiserType() const override { return DenoiserType::DLSS_DENOISER; }
    [[nodiscard]] const char* getDenoiserName() const override { return "DLSS Ray Reconstruction"; }
    void setOutputImage(sgl::vk::ImageViewPtr& outputImage) override;
    void setFeatureMap(FeatureMapType featureMapType, const sgl::vk::TexturePtr& featureTexture) override;
    [[nodiscard]] bool getUseFeatureMap(FeatureMapType featureMapType) const override;
    void setUseFeatureMap(FeatureMapType featureMapType, bool useFeature) override;
    void resetFrameNumber() override;
    void setTemporalDenoisingEnabled(bool enabled) override;
    void denoise() override;
    void recreateSwapchain(uint32_t width, uint32_t height) override;

#ifndef DISABLE_IMGUI
    /// Renders the GUI. Returns whether re-rendering has become necessary due to the user's actions.
    bool renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) override;
#endif
    void setSettings(const std::unordered_map<std::string, std::string>& settings) override;

private:
    bool initialize();
    void _free();
    void setPerfQuality(DlssPerfQuality _perfQuality);
    void setRenderPreset(DlssRenderPreset _preset);
    void setUpscaleAlpha(bool _upscaleAlpha);
    void setSharpeningFactor(float _sharpeningFactor);
    bool queryOptimalSettings(
            uint32_t displayWidth, uint32_t displayHeight,
            uint32_t& renderWidthOptimal, uint32_t& renderHeightOptimal,
            uint32_t& renderWidthMax, uint32_t& renderHeightMax,
            uint32_t& renderWidthMin, uint32_t& renderHeightMin,
            float& sharpness);
    void resetAccum();
    bool apply(
            const sgl::vk::ImageViewPtr& colorImageIn,
            const sgl::vk::ImageViewPtr& colorImageOut,
            const sgl::vk::ImageViewPtr& depthImage,
            const sgl::vk::ImageViewPtr& motionVectorImage,
            const sgl::vk::ImageViewPtr& exposureImage,
            const sgl::vk::ImageViewPtr& normalImage,
            const sgl::vk::ImageViewPtr& albedoImage,
            const sgl::vk::ImageViewPtr& roughnessImage,
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE);
    void checkRecreateFeature(
            const sgl::vk::ImageViewPtr& colorImageIn,
            const sgl::vk::ImageViewPtr& colorImageOut,
            VkCommandBuffer commandBuffer);

    sgl::vk::Renderer* renderer = nullptr;
    sgl::vk::ImageViewPtr inputImageVulkan, depthImageVulkan, normalImageVulkan, albedoImageVulkan, flowImageVulkan;
    sgl::vk::ImageViewPtr outputImageVulkan;
    sgl::vk::ImageViewPtr exposureImageView; // 1x1 image with constant content 1.0f
    sgl::vk::ImageViewPtr roughnessImageView; // Viewport-sized image with constant content 1.0f

    sgl::vk::Device* device = nullptr;
    bool isInitialized = false;
    NVSDK_NGX_Handle* dlssFeature = nullptr;
    NVSDK_NGX_Parameter* params = nullptr;
    DlssPerfQuality perfQuality = DlssPerfQuality::DLAA;
    DlssRenderPreset renderPreset = DlssRenderPreset::DEFAULT;
    bool enableTemporalAccumulation = true;
    bool shallResetAccum = true;
    uint32_t cachedRenderWidth = 0;
    uint32_t cachedRenderHeight = 0;
    uint32_t cachedDisplayWidth = 0;
    uint32_t cachedDisplayHeight = 0;
    // Settings of DLSS feature.
    bool motionVectorLowRes = true;
    bool isHdr = false;
    bool isDepthInverted = false;
    bool doSharpening = false;
    float sharpeningFactor = 0.0f;
    bool enableAutoExposure = false;
    bool upscaleAlpha = false;
    bool denoiseUnifiedOn = true;
};

#endif //CLOUDRENDERING_DLSSDENOISER_HPP

/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Christoph Neuhauser
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

#ifndef CLOUDRENDERING_DENOISER_HPP
#define CLOUDRENDERING_DENOISER_HPP

#include <string>
#include <unordered_map>

#include <Graphics/Vulkan/Image/Image.hpp>

namespace IGFD {
class FileDialog;
}
typedef IGFD::FileDialog ImGuiFileDialog;
namespace sgl {
class PropertyEditor;
}


enum class DenoiserType {
    NONE,
    EAW,
#ifdef SUPPORT_PYTORCH_DENOISER
    PYTORCH_DENOISER,
#endif
#ifdef SUPPORT_OPTIX
    OPTIX,
#endif
#ifdef SUPPORT_OPEN_IMAGE_DENOISE
    OPEN_IMAGE_DENOISE,
#endif
    SVGF
};
const char* const DENOISER_NAMES[] = {
        "None",
        "Edge-Avoiding À-Trous Wavelet Transform",
#ifdef SUPPORT_PYTORCH_DENOISER
        "PyTorch Denoiser Module",
#endif
#ifdef SUPPORT_OPTIX
        "OptiX Denoiser",
#endif
#ifdef SUPPORT_OPEN_IMAGE_DENOISE
        "OpenImageDenoise",
#endif
        "SVGF"
};


// enum type, name, num channels, num channels padded
#define FEATURE_MAPS                                          \
    FEATURE_MAP(COLOR,                "Color",                4, 4) \
    FEATURE_MAP(ALBEDO,               "Albedo",               4, 4) \
    FEATURE_MAP(FLOW,                 "Flow",                 2, 2) \
    FEATURE_MAP(POSITION,             "Position",             3, 4) \
    FEATURE_MAP(NORMAL,               "Normal",               3, 4) \
    FEATURE_MAP(NORMAL_LEN_1,         "Normal (Length 1)",    3, 4) \
    FEATURE_MAP(CLOUDONLY,            "CloudOnly",            4, 4) \
    FEATURE_MAP(DEPTH,                "Depth",                2, 2) \
    FEATURE_MAP(DENSITY,              "Density",              2, 2) \
    FEATURE_MAP(BACKGROUND,           "Background",           4, 4) \
    FEATURE_MAP(REPROJ_UV,            "Reproj_UV",            2, 2) \
    FEATURE_MAP(DEPTH_BLENDED,        "Depth Blended",        2, 2) \
    FEATURE_MAP(DEPTH_NEAREST_OPAQUE, "Depth Nearest Opaque", 2, 2) \
    FEATURE_MAP(DEPTH_NABLA,          "nabla(z)",             2, 2) \
    FEATURE_MAP(DEPTH_FWIDTH,         "fwidth(z)",            1, 1) \
    FEATURE_MAP(UNUSED,               "Unused",               2, 2) \

enum class FeatureMapType {
#define FEATURE_MAP(enum_name, _1, _2, _3) enum_name,
    FEATURE_MAPS
#undef FEATURE_MAP
};

const char* const FEATURE_MAP_NAMES[] = {
#define FEATURE_MAP(_1, string_name, _2, _3) string_name,
        FEATURE_MAPS
#undef FEATURE_MAP
};

const uint32_t FEATURE_MAP_NUM_CHANNELS[] = {
#define FEATURE_MAP(_1, _2, num_channels, _3) num_channels,
        FEATURE_MAPS
#undef FEATURE_MAP
};
const uint32_t FEATURE_MAP_NUM_CHANNELS_PADDED[] = {
#define FEATURE_MAP(_1, _2, _3, num_channels_padded) num_channels_padded,
        FEATURE_MAPS
#undef FEATURE_MAP
};


class Denoiser {
public:
    virtual ~Denoiser() = default;
    [[nodiscard]] virtual DenoiserType getDenoiserType() const = 0;
    [[nodiscard]] virtual const char* getDenoiserName() const = 0;
    [[nodiscard]] virtual bool getIsEnabled() const { return true; }
    virtual void setOutputImage(sgl::vk::ImageViewPtr& outputImage) = 0;
    virtual void setFeatureMap(FeatureMapType featureMapType, const sgl::vk::TexturePtr& featureTexture) = 0;
    [[nodiscard]] virtual bool getUseFeatureMap(FeatureMapType featureMapType) const = 0;
    [[nodiscard]] virtual bool getWantsAccumulatedInput() const { return true; }
    [[nodiscard]] virtual bool getWantsGlobalFrameNumber() const { return false; }
    virtual void setUseFeatureMap(FeatureMapType featureMapType, bool useFeature) = 0;
    virtual void setTemporalDenoisingEnabled(bool enabled) = 0;
    virtual void resetFrameNumber() = 0; // For temporal denoisers to indicate reset of temporal accumulation.
    virtual void denoise() = 0;
    virtual void recreateSwapchain(uint32_t width, uint32_t height) {}
    virtual void setFileDialogInstance(ImGuiFileDialog* _fileDialogInstance) {}
    virtual bool loadModelFromFile(const std::string& modelPath) { return false; }
    virtual void setOutputForegroundMap(bool _shallOutputForegroundMap) {}

    /// Renders the GUI. Returns whether re-rendering has become necessary due to the user's actions.
#ifndef DISABLE_IMGUI
    virtual bool renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) { return false; }
#endif
    virtual void setSettings(const std::unordered_map<std::string, std::string>& settings) {}
};

enum class DenoisingMode {
    PATH_TRACING, AMBIENT_OCCLUSION, VOLUMETRIC_PATH_TRACING
};

std::shared_ptr<Denoiser> createDenoiserObject(
        DenoiserType denoiserType, sgl::vk::Renderer* renderer, DenoisingMode mode, bool denoiseAlpha);

#endif //CLOUDRENDERING_DENOISER_HPP

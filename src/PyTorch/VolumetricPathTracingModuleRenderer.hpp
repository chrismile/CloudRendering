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

#ifndef CLOUDRENDERING_VOLUMETRICPATHTRACINGMODULERENDERER_HPP
#define CLOUDRENDERING_VOLUMETRICPATHTRACINGMODULERENDERER_HPP

#include <torch/types.h>

#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <Graphics/Vulkan/Utils/SyncObjects.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include "PathTracer/VolumetricPathTracingPass.hpp"

class CloudData;
typedef std::shared_ptr<CloudData> CloudDataPtr;
class VolumetricPathTracingPass;
enum class VptMode;
enum class GridInterpolationType;

namespace sgl {
class Camera;
typedef std::shared_ptr<Camera> CameraPtr;
namespace vk {
class BlitComputePass;
typedef std::shared_ptr<BlitComputePass> BlitComputePassPtr;
}
}

class ConvertTransmittanceVolumePass : public sgl::vk::ComputePass {
public:
    ConvertTransmittanceVolumePass(sgl::vk::Renderer* renderer);
    void setInputOutputData(const sgl::vk::ImageViewPtr& _inputImage, const sgl::vk::BufferPtr& _outputBuffer);
    void clearInputOutputData();

protected:
    void loadShader() override;
    void setComputePipelineInfo(sgl::vk::ComputePipelineInfo& pipelineInfo) override {}
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    const uint32_t BLOCK_SIZE_X = 8;
    const uint32_t BLOCK_SIZE_Y = 8;
    const uint32_t BLOCK_SIZE_Z = 4;
    sgl::vk::ImageViewPtr inputImage;
    sgl::vk::BufferPtr outputBuffer;
};

class VolumetricPathTracingModuleRenderer {
public:
    explicit VolumetricPathTracingModuleRenderer(sgl::vk::Renderer* renderer);
    ~VolumetricPathTracingModuleRenderer();

    inline sgl::MultiVarTransferFunctionWindow* getTransferFunctionWindow() { return transferFunctionWindow; }
    inline LightEditorWidget* getLightEditorWidget() { return lightEditorWidget; }

    /// Sets the cloud data that is rendered when calling @see renderFrameCpu.
    VolumetricPathTracingPass* getVptPass();
    const CloudDataPtr& getCloudData();
    void setCloudData(const CloudDataPtr& cloudData);
    void setEmissionData(const CloudDataPtr& cloudData);

    void loadEnvironmentMapImage(const std::string& filename);
    void setUseBuiltinEnvironmentMap(const std::string& envMapName);
    void setEnvironmentMapIntensityFactor(float intensityFactor);

    void setScatteringAlbedo(glm::vec3 albedo);
    void setExtinctionScale(double extinctionScale);
    void setExtinctionBase(glm::vec3 extinctionBase);
    void setPhaseG(double phaseG);
    void setUseFeatureMaps(const std::unordered_set<FeatureMapTypeVpt>& featureMapSet);
    void setFeatureMapType(FeatureMapTypeVpt type);
    void setDenoiserType(DenoiserType _denoiserType);
    void setDenoiserProperty(const std::string& key, const std::string& value);
    void checkDenoiser();

    void setEmissionCap(double emissionCap);
    void setEmissionStrength(double emissionStrength);
    void setUseEmission(bool useEmission);
    void flipYZ(bool flip);

    const glm::vec3& getCameraPosition();
    const sgl::CameraPtr& getCamera();

    void setCameraPosition(const glm::vec3& cameraPosition);
    void setCameraTarget(const glm::vec3& cameraTarget);
    void setCameraFOVy(double FOVy);

    void setGlobalWorldBoundingBox(const sgl::AABB3& boundingBox);
    [[nodiscard]] const sgl::AABB3& getGlobalWorldBoundingBox() const { return globalWorldBoundingBox; }
    [[nodiscard]] bool getHasGlobalWorldBoundingBox() const { return hasGlobalWorldBoundingBox; }
    void rememberNextBounds();
    void forgetCurrentBounds();

    void setViewProjectionMatrixAsPrevious();

    /// Called when the resolution of the application window has changed.
    void setRenderingResolution(
            uint32_t width, uint32_t height, uint32_t channels, c10::Device torchDevice, caffe2::TypeMeta dtype);
    [[nodiscard]] bool settingsDiffer(
            uint32_t width, uint32_t height, uint32_t channels, c10::Device torchDevice, caffe2::TypeMeta dtype) const;
    [[nodiscard]] inline bool getHasFrameData() const { return renderImageView.get() != nullptr; }
    [[nodiscard]] inline uint32_t getFrameWidth() const { return renderImageView->getImage()->getImageSettings().width; }
    [[nodiscard]] inline uint32_t getFrameHeight() const { return renderImageView->getImage()->getImageSettings().height; }
    [[nodiscard]] inline uint32_t getNumChannels() const { return numChannels; }
    [[nodiscard]] inline caffe2::TypeMeta getDType() const { return dtype; }
    [[nodiscard]] inline c10::DeviceType getDeviceType() const { return deviceType; }

    /// Sets whether a dense or sparse grid should be used.
    void setUseSparseGrid(bool useSparseGrid);
    void setGridInterpolationType(GridInterpolationType type);

    /// Sets an additive offset for the random seed in the VPT shader.
    void setCustomSeedOffset(uint32_t offset);

    /// Sets whether linear RGB or sRGB should be used for rendering.
    void setUseLinearRGB(bool useLinearRGB);

    /// Sets the volumetric path tracing mode used for rendering.
    void setVptMode(VptMode vptMode);
    void setVptModeFromString(const std::string& vptModeName);

    /**
     * Renders the path traced volume object to the scene framebuffer and returns a CPU pointer to the image data.
     * @param numFrames The number of frames to accumulate.
     * @return A CPU floating point array of size width * height * 3 containing the frame data.
     * NOTE: The returned data is managed by this class.
     */
    float* renderFrameCpu(uint32_t numFrames);

    float* renderFrameVulkan(uint32_t numFrames);

    float* getFeatureMapCpu(FeatureMapTypeVpt featureMap);
    float* getFeatureMapCuda(FeatureMapTypeVpt featureMap);

    float* computeOccupationVolumeCuda(uint32_t ds, uint32_t maxKernelRadius);

#ifdef SUPPORT_CUDA_INTEROP
    void createCommandStructures(uint32_t numFrames);
    float* renderFrameCuda(uint32_t numFrames);
#endif

private:
    sgl::MultiVarTransferFunctionWindow* transferFunctionWindow;
    LightEditorWidget* lightEditorWidget = nullptr;
    sgl::CameraPtr camera;
    glm::vec3 cameraPosition = glm::vec3(0,0,0);
    glm::vec3 cameraTarget = glm::vec3(0,0,0);

    glm::vec3 seq_bounds_min = glm::vec3(0,0,0);
    glm::vec3 seq_bounds_max = glm::vec3(0,0,0);
    bool hasStoredBounds = false;
    bool storeNextBounds = false;

    sgl::AABB3 globalWorldBoundingBox{};
    bool hasGlobalWorldBoundingBox = false;

    glm::mat4 previousViewProjectionMatrix = glm::zero<glm::mat4>();

    sgl::vk::Renderer* renderer = nullptr;
    std::shared_ptr<VolumetricPathTracingPass> vptPass;

    // Copy pass for 3D volume data.
    std::shared_ptr<ConvertTransmittanceVolumePass> convertTransmittanceVolumePass;

    DenoiserType denoiserType = DenoiserType::NONE;
    std::unordered_map<std::string, std::string> denoiserSettings;
    bool isDenoiserDirty = false;

    sgl::vk::ImageViewPtr renderImageView;
    sgl::vk::BlitComputePassPtr renderImageBlitPass;
    uint32_t numChannels = 0;
    caffe2::TypeMeta dtype;
    c10::DeviceType deviceType;

    // Data for CPU rendering.
    sgl::vk::ImagePtr renderImageStaging;
    sgl::vk::FencePtr renderFinishedFence;
    float* imageData = nullptr;

    // Data for Vulkan rendering.

#ifdef SUPPORT_CUDA_INTEROP
    // Data for CUDA rendering.
    // Image data.
    sgl::vk::BufferPtr outputImageBufferVk;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr outputImageBufferCu;
    sgl::vk::BufferPtr outputVolumeBufferVk;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr outputVolumeBufferCu;
    sgl::vk::BufferPtr outputOccupationVolumeBufferVk;
    sgl::vk::BufferCudaDriverApiExternalMemoryVkPtr outputOccupationVolumeBufferCu;
    // Synchronization primitives.
    const uint32_t maxNumFramesInFlight = 32;
    std::vector<sgl::vk::CommandBufferPtr> commandBuffers;
    std::vector<sgl::vk::FencePtr> frameFences;
    sgl::vk::SemaphoreVkCudaDriverApiInteropPtr renderReadySemaphore;
    sgl::vk::SemaphoreVkCudaDriverApiInteropPtr renderFinishedSemaphore;
    std::vector<sgl::vk::SemaphorePtr> interFrameSemaphores;
    uint64_t timelineValue = 0;
#ifdef SUPPORT_OPTIX
    bool optixInitialized = false;
#endif
#endif
};

#endif //CLOUDRENDERING_VOLUMETRICPATHTRACINGMODULERENDERER_HPP

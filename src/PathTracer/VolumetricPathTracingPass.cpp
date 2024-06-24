/**
 * MIT License
 *
 * Copyright (c) 2021, Christoph Neuhauser, Timm Knörle, Ludwig Leonard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <memory>
#include <utility>
#include <glm/vec3.hpp>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Math/Math.hpp>
#include <Math/half/half.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Texture/Bitmap.hpp>
#include <Graphics/Vulkan/Buffers/Framebuffer.hpp>
#include <Graphics/Vulkan/Render/RayTracingPipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#include <ImGui/Widgets/MultiVarTransferFunctionWindow.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>
#include <ImGui/imgui_stdlib.h>

#include "Denoiser/EAWDenoiser.hpp"
#ifdef SUPPORT_OPTIX
#include "Denoiser/OptixVptDenoiser.hpp"
#endif

#ifdef SUPPORT_OPENEXR
#include "OpenExrLoader.hpp"
#endif

#include "Utils/CameraPoseLinePass.hpp"
#include "CloudData.hpp"
#include "MomentUtils.hpp"
#include "SuperVoxelGrid.hpp"
#include "VolumetricPathTracingPass.hpp"

VolumetricPathTracingPass::VolumetricPathTracingPass(sgl::vk::Renderer* renderer, sgl::CameraPtr* camera)
        : ComputePass(renderer), camera(camera) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    frameInfoBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(FrameInfo),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    computeWrappingZoneParameters(momentUniformData.wrapping_zone_parameters);
    momentUniformDataBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(MomentUniformData), &momentUniformData,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    equalAreaPass = std::make_shared<OctahedralMappingPass>(renderer);

    if (sgl::AppSettings::get()->getSettings().getValueOpt(
            "vptEnvironmentMapImage", environmentMapFilenameGui)) {
        useEnvironmentMapImage = true;
        sgl::AppSettings::get()->getSettings().getValueOpt("vptUseEnvironmentMap", useEnvironmentMapImage);
        loadEnvironmentMapImage(environmentMapFilenameGui);
    }

    blitResultRenderPass = std::make_shared<sgl::vk::BlitRenderPass>(renderer);
    blitPrimaryRayMomentTexturePass = std::make_shared<BlitMomentTexturePass>(renderer, "Primary");
    blitScatterRayMomentTexturePass = std::make_shared<BlitMomentTexturePass>(renderer, "Scatter");
    cameraPoseLinePass = std::make_shared<CameraPoseLinePass>(renderer);

    createDenoiser();
    updateVptMode();
}

VolumetricPathTracingPass::~VolumetricPathTracingPass() {
    if (isEnvironmentMapLoaded) {
        sgl::AppSettings::get()->getSettings().addKeyValue(
                "vptEnvironmentMapImage", loadedEnvironmentMapFilename);
        sgl::AppSettings::get()->getSettings().addKeyValue(
                "vptUseEnvironmentMap", useEnvironmentMapImage);
    }
}

void VolumetricPathTracingPass::createDenoiser() {
    denoiser = createDenoiserObject(denoiserType, renderer, DenoisingMode::VOLUMETRIC_PATH_TRACING, denoiseAlpha);
    if (denoiser) {
        denoiser->setFileDialogInstance(fileDialogInstance);
    }

    globalFrameNumber = 0;
    if (resultImageTexture) {
        checkRecreateFeatureMaps();
        setDenoiserFeatureMaps();
        if (denoiser) {
            denoiser->recreateSwapchain(lastViewportWidth, lastViewportHeight);
        }
    }
}

void VolumetricPathTracingPass::setOutputImage(sgl::vk::ImageViewPtr& imageView) {
    sceneImageView = imageView;

    sgl::vk::ImageSamplerSettings samplerSettings;
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    sgl::vk::ImageSettings imageSettings = imageView->getImage()->getImageSettings();
    imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    resultImageView = std::make_shared<sgl::vk::ImageView>(
            std::make_shared<sgl::vk::Image>(device, imageSettings));
    resultTexture = std::make_shared<sgl::vk::Texture>(
            resultImageView, sgl::vk::ImageSamplerSettings());
    imageSettings.usage =
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    denoisedImageView = std::make_shared<sgl::vk::ImageView>(
            std::make_shared<sgl::vk::Image>(device, imageSettings));

    resultImageTexture = std::make_shared<sgl::vk::Texture>(resultImageView, samplerSettings);

    cameraPoseLinePass->setRenderTarget(sceneImageView);
    cameraPoseLinePass->recreateSwapchain(imageSettings.width, imageSettings.height);

    imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    accImageTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    /*firstXTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    firstWTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    normalTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    cloudOnlyTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    backgroundTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);

    imageSettings.format = VK_FORMAT_R32G32_SFLOAT;
    depthTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    densityTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    reprojUVTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    setDenoiserFeatureMaps();*/

    blitResultRenderPass->setInputTexture(resultTexture);
    blitResultRenderPass->setOutputImage(imageView);
    blitPrimaryRayMomentTexturePass->setOutputImage(imageView);
    blitScatterRayMomentTexturePass->setOutputImage(imageView);


    frameInfo.frameCount = 0;
    setDataDirty();
}

void VolumetricPathTracingPass::recreateSwapchain(uint32_t width, uint32_t height) {
    lastViewportWidth = width;
    lastViewportHeight = height;

    recreateFeatureMaps();

    blitResultRenderPass->recreateSwapchain(width, height);
    blitScatterRayMomentTexturePass->recreateSwapchain(width, height);
    blitPrimaryRayMomentTexturePass->recreateSwapchain(width, height);

    if (useDenoiser && denoiser) {
        denoiser->recreateSwapchain(width, height);
    }
}

void VolumetricPathTracingPass::setDenoiserFeatureMaps() {
    if (denoiser) {
        if (denoiser->getUseFeatureMap(FeatureMapType::COLOR)) {
            denoiser->setFeatureMap(FeatureMapType::COLOR, resultImageTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::POSITION)) {
            denoiser->setFeatureMap(FeatureMapType::POSITION, firstXTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::NORMAL)) {
            denoiser->setFeatureMap(FeatureMapType::NORMAL, normalTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::DEPTH)) {
            denoiser->setFeatureMap(FeatureMapType::DEPTH, depthTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::DENSITY)) {
            denoiser->setFeatureMap(FeatureMapType::DENSITY, densityTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::CLOUDONLY)) {
            denoiser->setFeatureMap(FeatureMapType::CLOUDONLY, cloudOnlyTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::BACKGROUND)) {
            denoiser->setFeatureMap(FeatureMapType::BACKGROUND, backgroundTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::REPROJ_UV)) {
            denoiser->setFeatureMap(FeatureMapType::REPROJ_UV, reprojUVTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::DEPTH_BLENDED)) {
            denoiser->setFeatureMap(FeatureMapType::DEPTH_BLENDED, depthBlendedTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::FLOW)) {
            denoiser->setFeatureMap(FeatureMapType::FLOW, flowTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::DEPTH_NABLA)) {
            denoiser->setFeatureMap(FeatureMapType::DEPTH_NABLA, depthNablaTexture);
        }
        if (denoiser->getUseFeatureMap(FeatureMapType::DEPTH_FWIDTH)) {
            denoiser->setFeatureMap(FeatureMapType::DEPTH_FWIDTH, depthFwidthTexture);
        }

        denoiser->setOutputImage(denoisedImageView);

        featureMapUsedArray.resize(IM_ARRAYSIZE(FEATURE_MAP_NAMES));
        for (int i = 0; i < IM_ARRAYSIZE(FEATURE_MAP_NAMES); i++) {
            featureMapUsedArray.at(i) = denoiser->getUseFeatureMap(FeatureMapType(i));
        }
    }
}

void VolumetricPathTracingPass::recreateFeatureMaps() {
    sgl::vk::ImageSamplerSettings samplerSettings;
    sgl::vk::ImageSettings imageSettings;
    imageSettings.width = lastViewportWidth;
    imageSettings.height = lastViewportHeight;

    firstXTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::FIRST_X)))
            || featureMapType == FeatureMapTypeVpt::FIRST_X || featureMapSet.find(FeatureMapTypeVpt::FIRST_X) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        firstXTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    firstWTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::FIRST_W)))
            || featureMapType == FeatureMapTypeVpt::FIRST_W || featureMapSet.find(FeatureMapTypeVpt::FIRST_W) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        firstWTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    normalTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::NORMAL)))
            || featureMapType == FeatureMapTypeVpt::NORMAL || featureMapSet.find(FeatureMapTypeVpt::NORMAL) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        normalTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    cloudOnlyTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::CLOUD_ONLY)))
            || featureMapType == FeatureMapTypeVpt::CLOUD_ONLY || featureMapSet.find(FeatureMapTypeVpt::CLOUD_ONLY) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        cloudOnlyTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    backgroundTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::BACKGROUND)))
            || featureMapType == FeatureMapTypeVpt::BACKGROUND || featureMapSet.find(FeatureMapTypeVpt::BACKGROUND) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        backgroundTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    depthTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DEPTH)))
            || featureMapType == FeatureMapTypeVpt::DEPTH || featureMapSet.find(FeatureMapTypeVpt::DEPTH) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        depthTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    densityTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DENSITY)))
            || featureMapType == FeatureMapTypeVpt::DENSITY || featureMapSet.find(FeatureMapTypeVpt::DENSITY) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        densityTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    reprojUVTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::REPROJ_UV)))
            || featureMapType == FeatureMapTypeVpt::REPROJ_UV || featureMapSet.find(FeatureMapTypeVpt::REPROJ_UV) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        reprojUVTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    depthBlendedTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DEPTH_BLENDED)))
            || featureMapType == FeatureMapTypeVpt::DEPTH_BLENDED || featureMapSet.find(FeatureMapTypeVpt::DEPTH_BLENDED) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        depthBlendedTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    flowTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::FLOW)))
            || featureMapType == FeatureMapTypeVpt::FLOW || featureMapSet.find(FeatureMapTypeVpt::FLOW) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        flowTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    depthNablaTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DEPTH_NABLA)))
            || featureMapType == FeatureMapTypeVpt::DEPTH_NABLA || featureMapSet.find(FeatureMapTypeVpt::DEPTH_NABLA) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32G32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        depthNablaTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    depthFwidthTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DEPTH_FWIDTH)))
        || featureMapType == FeatureMapTypeVpt::DEPTH_FWIDTH || featureMapSet.find(FeatureMapTypeVpt::DEPTH_FWIDTH) != featureMapSet.end()) {
        imageSettings.format = VK_FORMAT_R32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        depthFwidthTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    }

    transmittanceVolumeTexture = {};
    if ((denoiser && denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::TRANSMITTANCE_VOLUME)))
            || featureMapType == FeatureMapTypeVpt::TRANSMITTANCE_VOLUME || featureMapSet.find(FeatureMapTypeVpt::TRANSMITTANCE_VOLUME) != featureMapSet.end()) {
        sgl::vk::ImageSettings imageSettings3d;
        imageSettings3d.width = sgl::uiceil(cloudData->getGridSizeX(), dsSecondaryVolume);
        imageSettings3d.height = sgl::uiceil(cloudData->getGridSizeY(), dsSecondaryVolume);
        imageSettings3d.depth = sgl::uiceil(cloudData->getGridSizeZ(), dsSecondaryVolume);
        imageSettings3d.imageType = VK_IMAGE_TYPE_3D;
        imageSettings3d.format = VK_FORMAT_R32_UINT;
        imageSettings3d.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        transmittanceVolumeTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings3d, samplerSettings);
    }

    /*albedoTexture = {};
    if (denoiser && denoiser->getUseFeatureMap(FeatureMapType::ALBEDO)) {
        imageSettings.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        imageSettings.usage =
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        albedoTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
        VkCommandBuffer commandBuffer = device->beginSingleTimeCommands();
        albedoTexture->getImage()->transitionImageLayout(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, commandBuffer);
        albedoTexture->getImageView()->clearColor(glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), commandBuffer);
        device->endSingleTimeCommands(commandBuffer);
    }*/

    setDenoiserFeatureMaps();
}

void VolumetricPathTracingPass::checkRecreateFeatureMaps() {
    bool useFirstXRenderer = firstXTexture.get() != nullptr;
    bool useFirstWRenderer = firstWTexture.get() != nullptr;
    bool useNormalRenderer = normalTexture.get() != nullptr;
    bool useCloudOnlyRenderer = cloudOnlyTexture.get() != nullptr;
    bool useBackgroundRenderer = backgroundTexture.get() != nullptr;
    bool useDepthRenderer = depthTexture.get() != nullptr;
    bool useDensityRenderer = densityTexture.get() != nullptr;
    bool useReprojUVRenderer = reprojUVTexture.get() != nullptr;
    bool useDepthBlendedRenderer = depthBlendedTexture.get() != nullptr;
    bool useFlowRenderer = flowTexture.get() != nullptr;
    bool useDepthNablaRenderer = depthNablaTexture.get() != nullptr;
    bool useDepthFwidthRenderer = depthFwidthTexture.get() != nullptr;
    bool useTransmittanceVolumeRenderer = transmittanceVolumeTexture.get() != nullptr;
    //bool useAlbedoRenderer = albedoTexture.get() != nullptr;

    bool shallRecreateFeatureMaps = false;
    if (denoiser) {
        if (useFirstXRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::FIRST_X))
                    || featureMapType == FeatureMapTypeVpt::FIRST_X || featureMapSet.find(FeatureMapTypeVpt::FIRST_X) != featureMapSet.end())
                || useFirstWRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::FIRST_W))
                    || featureMapType == FeatureMapTypeVpt::FIRST_W || featureMapSet.find(FeatureMapTypeVpt::FIRST_W) != featureMapSet.end())
                || useNormalRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::NORMAL))
                    || featureMapType == FeatureMapTypeVpt::NORMAL || featureMapSet.find(FeatureMapTypeVpt::NORMAL) != featureMapSet.end())
                || useCloudOnlyRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::CLOUD_ONLY))
                    || featureMapType == FeatureMapTypeVpt::CLOUD_ONLY || featureMapSet.find(FeatureMapTypeVpt::CLOUD_ONLY) != featureMapSet.end())
                || useBackgroundRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::BACKGROUND))
                    || featureMapType == FeatureMapTypeVpt::BACKGROUND || featureMapSet.find(FeatureMapTypeVpt::BACKGROUND) != featureMapSet.end())
                || useDepthRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DEPTH))
                    || featureMapType == FeatureMapTypeVpt::DEPTH || featureMapSet.find(FeatureMapTypeVpt::DEPTH) != featureMapSet.end())
                || useDensityRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DENSITY))
                    || featureMapType == FeatureMapTypeVpt::DENSITY || featureMapSet.find(FeatureMapTypeVpt::DENSITY) != featureMapSet.end())
                || useReprojUVRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::REPROJ_UV))
                    || featureMapType == FeatureMapTypeVpt::REPROJ_UV || featureMapSet.find(FeatureMapTypeVpt::REPROJ_UV) != featureMapSet.end())
                || useDepthBlendedRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DEPTH_BLENDED))
                    || featureMapType == FeatureMapTypeVpt::DEPTH_BLENDED || featureMapSet.find(FeatureMapTypeVpt::DEPTH_BLENDED) != featureMapSet.end())
                || useFlowRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::FLOW))
                    || featureMapType == FeatureMapTypeVpt::FLOW || featureMapSet.find(FeatureMapTypeVpt::FLOW) != featureMapSet.end())
                || useDepthNablaRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DEPTH_NABLA))
                    || featureMapType == FeatureMapTypeVpt::DEPTH_NABLA || featureMapSet.find(FeatureMapTypeVpt::DEPTH_NABLA) != featureMapSet.end())
                || useDepthFwidthRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::DEPTH_FWIDTH))
                    || featureMapType == FeatureMapTypeVpt::DEPTH_FWIDTH || featureMapSet.find(FeatureMapTypeVpt::DEPTH_FWIDTH) != featureMapSet.end())
                || useTransmittanceVolumeRenderer != (denoiser->getUseFeatureMap(featureMapCorrespondence.getCorrespondenceDenoiser(FeatureMapTypeVpt::TRANSMITTANCE_VOLUME))
                    || featureMapType == FeatureMapTypeVpt::TRANSMITTANCE_VOLUME || featureMapSet.find(FeatureMapTypeVpt::TRANSMITTANCE_VOLUME) != featureMapSet.end())
        ) {
            shallRecreateFeatureMaps = true;
        }
    } else {
        if (useFirstXRenderer || useFirstWRenderer || useNormalRenderer
                || useCloudOnlyRenderer || useBackgroundRenderer || useDepthRenderer
                || useDensityRenderer || useReprojUVRenderer || useDepthBlendedRenderer
                || useTransmittanceVolumeRenderer) {
            shallRecreateFeatureMaps = true;
        }
    }

    // Check if inputs should be accumulated.
    if (denoiser) {
        if (accumulateInputs != denoiser->getWantsAccumulatedInput()) {
            accumulateInputs = denoiser->getWantsAccumulatedInput();
            shallRecreateFeatureMaps = true;
        }
        useGlobalFrameNumber = denoiser->getWantsGlobalFrameNumber();
    } else {
        if (!accumulateInputs) {
            accumulateInputs = true;
            shallRecreateFeatureMaps = true;
        }
        useGlobalFrameNumber = false;
    }

    if (shallRecreateFeatureMaps) {
        setShaderDirty();
        device->waitIdle();
        recreateFeatureMaps();
        //onHasMovedParent();
        changedDenoiserSettings = false;
    }
}

/*void VolumetricPathTracingPass::checkResetDenoiserFeatureMaps() {
    bool shallResetFeatureMaps = false;
    if (denoiser) {
        for (int i = 0; i < IM_ARRAYSIZE(FEATURE_MAP_NAMES); i++) {
            if (denoiser->getUseFeatureMap(FeatureMapType(i)) != featureMapUsedArray.at(i)) {
                shallResetFeatureMaps = true;
            }
        }
    }

    if (shallResetFeatureMaps) {
        setDenoiserFeatureMaps();
        //changedDenoiserSettings = false;
    }
}*/

void VolumetricPathTracingPass::setGridData() {
    nanoVdbBuffer = {};
    densityFieldTexture = {};
    emissionNanoVdbBuffer = {};
    emissionFieldTexture = {};
    densityGradientFieldTexture = {};

    if (!cloudData) {
        return;
    }

    if (useSparseGrid) {
        uint8_t* sparseDensityField;
        uint64_t sparseDensityFieldSize;
        cloudData->getSparseDensityField(sparseDensityField, sparseDensityFieldSize);

        uint64_t bufferSize = sizeof(uint32_t) * sgl::ulceil(sparseDensityFieldSize, sizeof(uint32_t));
        auto* sparseDensityFieldCopy = new uint8_t[bufferSize];
        memset(sparseDensityFieldCopy, 0, bufferSize);
        memcpy(sparseDensityFieldCopy, sparseDensityField, sparseDensityFieldSize);

        nanoVdbBuffer = std::make_shared<sgl::vk::Buffer>(
                device, bufferSize, sparseDensityFieldCopy,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
        delete[] sparseDensityFieldCopy;

        /*if (emissionData && useEmission) {
            emissionData->getSparseDensityField(sparseDensityField, sparseDensityFieldSize);

            bufferSize = sizeof(uint32_t) * sgl::iceil(int(sparseDensityFieldSize), sizeof(uint32_t));
            sparseDensityFieldCopy = new uint8_t[bufferSize];
            memset(sparseDensityFieldCopy, 0, bufferSize);
            memcpy(sparseDensityFieldCopy, sparseDensityField, sparseDensityFieldSize);

            nanoVdbBuffer = std::make_shared<sgl::vk::Buffer>(
                    device, bufferSize, sparseDensityFieldCopy,
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_GPU_ONLY);
            delete[] sparseDensityFieldCopy;
        }*/
    } else {
        sgl::vk::ImageSettings imageSettings;
        imageSettings.width = cloudData->getGridSizeX();
        imageSettings.height = cloudData->getGridSizeY();
        imageSettings.depth = cloudData->getGridSizeZ();
        imageSettings.imageType = VK_IMAGE_TYPE_3D;
        imageSettings.format = cloudData->getDenseDensityField()->getEntryVulkanFormat();
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

        sgl::vk::ImageSamplerSettings samplerSettings;
        if (clampToZeroBorder) {
            samplerSettings.addressModeU = samplerSettings.addressModeV = samplerSettings.addressModeW =
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        } else {
            samplerSettings.addressModeU = samplerSettings.addressModeV = samplerSettings.addressModeW =
                    VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }
        samplerSettings.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        if (gridInterpolationType == GridInterpolationType::TRILINEAR) {
            samplerSettings.minFilter = VK_FILTER_LINEAR;
            samplerSettings.magFilter = VK_FILTER_LINEAR;
        } else {
            samplerSettings.minFilter = VK_FILTER_NEAREST;
            samplerSettings.magFilter = VK_FILTER_NEAREST;
        }
        sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

        densityFieldTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
        size_t numEntriesDensityGrid =
                size_t(cloudData->getGridSizeX()) * size_t(cloudData->getGridSizeY()) * size_t(cloudData->getGridSizeZ());
        densityFieldTexture->getImage()->uploadData(
                numEntriesDensityGrid * cloudData->getDenseDensityField()->getEntrySizeInBytes(),
                cloudData->getDenseDensityField()->getDataNative());

        if (emissionData && useEmission) {
            sgl::vk::ImageSettings emissionImageSettings;
            emissionImageSettings.width = emissionData->getGridSizeX();
            emissionImageSettings.height = emissionData->getGridSizeY();
            emissionImageSettings.depth = emissionData->getGridSizeZ();
            emissionImageSettings.imageType = VK_IMAGE_TYPE_3D;
            emissionImageSettings.format = emissionData->getDenseDensityField()->getEntryVulkanFormat();
            emissionImageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            emissionFieldTexture = std::make_shared<sgl::vk::Texture>(device, emissionImageSettings, samplerSettings);
            size_t numEntriesEmissionGrid =
                    size_t(emissionData->getGridSizeX()) * size_t(emissionData->getGridSizeY()) * size_t(emissionData->getGridSizeZ());
            emissionFieldTexture->getImage()->uploadData(
                    numEntriesEmissionGrid * emissionData->getDenseDensityField()->getEntrySizeInBytes(),
                    emissionData->getDenseDensityField()->getDataNative());
        }

        if (useIsosurfaces && isosurfaceType == IsosurfaceType::GRADIENT) {
            auto xs = size_t(cloudData->getGridSizeX());
            auto ys = size_t(cloudData->getGridSizeY());
            auto zs = size_t(cloudData->getGridSizeZ());
            auto df = cloudData->getDenseDensityField();
            auto numEntries = xs * ys * zs;

#define IDXS(x,y,z) ((z)*xs*ys + (y)*xs + (x))
            DensityFieldPtr gradField;
            if (numEntries >= size_t(1ull << 31ull)) {
                gradField = DensityField::createHalfFloat(numEntries, new HalfFloat[numEntries]);
            } else {
                gradField = std::make_shared<DensityField>(numEntries, new float[numEntries]);
            }
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<size_t>(0, numEntries), [&](auto const& r) {
                for (auto gridIdx = r.begin(); gridIdx != r.end(); gridIdx++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for default(none) shared(xs, ys, zs, numEntries, df, gradField)
#endif
            for (size_t gridIdx = 0; gridIdx < numEntries; gridIdx++) {
#endif
                auto xc = gridIdx % xs;
                auto yc = (gridIdx / xs) % ys;
                auto zc = gridIdx / (xs * ys);
                auto xl = xc > 0 ? xc - 1 : 0;
                auto yl = yc > 0 ? yc - 1 : 0;
                auto zl = zc > 0 ? zc - 1 : 0;
                auto xu = xc < xs - 1 ? xc + 1 : xs - 1;
                auto yu = yc < ys - 1 ? yc + 1 : ys - 1;
                auto zu = zc < zs - 1 ? zc + 1 : zs - 1;
                float dx = (df->getDataFloatAt(IDXS(xu, yc, zc)) - df->getDataFloatAt(IDXS(xl, yc, zc))) / float(xu - xl);
                float dy = (df->getDataFloatAt(IDXS(xc, yu, zc)) - df->getDataFloatAt(IDXS(xc, yl, zc))) / float(yu - yl);
                float dz = (df->getDataFloatAt(IDXS(xc, yc, zu)) - df->getDataFloatAt(IDXS(xc, yc, zl))) / float(zu - zl);
                float gradVal = std::sqrt(dx * dx + dy * dy + dz * dz);
                gradField->setDataFloatAt(gridIdx, gradVal);
            }
#ifdef USE_TBB
            });
#endif
            minGradientVal = gradField->getMinValue();
            maxGradientVal = gradField->getMaxValue();

            sgl::vk::ImageSettings gradImageSettings;
            gradImageSettings.width = cloudData->getGridSizeX();
            gradImageSettings.height = cloudData->getGridSizeY();
            gradImageSettings.depth = cloudData->getGridSizeZ();
            gradImageSettings.imageType = VK_IMAGE_TYPE_3D;
            gradImageSettings.format = gradField->getEntryVulkanFormat();
            gradImageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            densityGradientFieldTexture = std::make_shared<sgl::vk::Texture>(device, gradImageSettings, samplerSettings);
            densityGradientFieldTexture->getImage()->uploadData(
                    numEntries * gradField->getEntrySizeInBytes(), gradField->getDataNative());
        }
    }
}

void VolumetricPathTracingPass::updateGridSampler() {
    if (!densityFieldTexture) {
        return;
    }

    sgl::vk::ImageSamplerSettings samplerSettings = densityFieldTexture->getImageSampler()->getImageSamplerSettings();
    if (clampToZeroBorder) {
        samplerSettings.addressModeU = samplerSettings.addressModeV = samplerSettings.addressModeW =
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    } else {
        samplerSettings.addressModeU = samplerSettings.addressModeV = samplerSettings.addressModeW =
                VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
    if (gridInterpolationType == GridInterpolationType::TRILINEAR) {
        samplerSettings.minFilter = VK_FILTER_LINEAR;
        samplerSettings.magFilter = VK_FILTER_LINEAR;
    } else {
        samplerSettings.minFilter = VK_FILTER_NEAREST;
        samplerSettings.magFilter = VK_FILTER_NEAREST;
    }
    densityFieldTexture = std::make_shared<sgl::vk::Texture>(
            densityFieldTexture->getImageView(), samplerSettings);
}

const CloudDataPtr& VolumetricPathTracingPass::getCloudData() {
    return cloudData;
}

void VolumetricPathTracingPass::setCloudData(const CloudDataPtr& data) {
    cloudData = data;
    frameInfo.frameCount = 0;
    globalFrameNumber = 0;

    auto numGridEntries =
            size_t(cloudData->getGridSizeX()) * size_t(cloudData->getGridSizeY()) * size_t(cloudData->getGridSizeZ());
    // For data larger than 4GB, default to sparse data if possible.
    if (cloudData->hasSparseData()) {
        auto gridSizeSparse = cloudData->getSparseDataSizeInBytes();
        // The NanoVDB GLSL implementation seems to have a problem with data >= 2GiB.
        if (numGridEntries >= size_t(1024) * size_t(1024) * size_t(1024)
                && gridSizeSparse < size_t(2) * size_t(1024) * size_t(1024) * size_t(1024)) {
            useSparseGrid = true;
        }
    }

    setGridData();
    setDataDirty();
    updateVptMode();
}

void VolumetricPathTracingPass::setEmissionData(const CloudDataPtr& data) {
    emissionData = data;
    frameInfo.frameCount = 0;

    setGridData();
    setDataDirty();
    updateVptMode();
}

void VolumetricPathTracingPass::setVptMode(VptMode vptMode) {
    this->vptMode = vptMode;
    if (vptMode == VptMode::ISOSURFACE_RENDERING && gridInterpolationType != GridInterpolationType::TRILINEAR) {
        updateGridSampler();
    }
    updateVptMode();
    setShaderDirty();
    setDataDirty();
}

void VolumetricPathTracingPass::setUseSparseGrid(bool useSparse) {
    this->useSparseGrid = useSparse;
    setGridData();
    updateVptMode();
    setShaderDirty();
    setDataDirty();
}

void VolumetricPathTracingPass::setSparseGridInterpolationType(GridInterpolationType type) {
    this->gridInterpolationType = type;
    updateGridSampler();
    setShaderDirty();
}

void VolumetricPathTracingPass::setCustomSeedOffset(uint32_t offset) {
    customSeedOffset = offset;
    setShaderDirty();
}

void VolumetricPathTracingPass::setUseLinearRGB(bool useLinearRGB) {
    uniformData.useLinearRGB = useLinearRGB;
    frameInfo.frameCount = 0;
    setShaderDirty();
}

void VolumetricPathTracingPass::setPreviousViewProjMatrix(glm::mat4 previousViewProjectionMatrix) {
    this->previousViewProjMatrix = previousViewProjectionMatrix;
}

void VolumetricPathTracingPass::setUseEmission(bool emission){
    useEmission = emission;
    frameInfo.frameCount = 0;
}

void VolumetricPathTracingPass::setEmissionStrength(float emissionStrength){
    this->emissionStrength = emissionStrength;
    frameInfo.frameCount = 0;
}

void VolumetricPathTracingPass::setEmissionCap(float emissionCap){
    this->emissionCap = emissionCap;
    frameInfo.frameCount = 0;
}

void VolumetricPathTracingPass::flipYZ(bool flip) {
    this->flipYZCoordinates = flip;
    frameInfo.frameCount = 0;
}

void VolumetricPathTracingPass::setUseIsosurfaces(bool _useIsosurfaces) {
    if (vptMode == VptMode::ISOSURFACE_RENDERING && !_useIsosurfaces) {
        _useIsosurfaces = true;
    }
    if (useIsosurfaces != _useIsosurfaces) {
        useIsosurfaces = _useIsosurfaces;
        if (gridInterpolationType != GridInterpolationType::TRILINEAR) {
            gridInterpolationType = GridInterpolationType::TRILINEAR;
            updateGridSampler();
        }
        setShaderDirty();
        if (isosurfaceType == IsosurfaceType::GRADIENT) {
            setGridData();
        }
        reRender = true;
        frameInfo.frameCount = 0;
    }
}

void VolumetricPathTracingPass::setIsoValue(float _isoValue) {
    if (isoValue != _isoValue) {
        isoValue = _isoValue;
        reRender = true;
        frameInfo.frameCount = 0;
    }
}

void VolumetricPathTracingPass::setIsoSurfaceColor(const glm::vec3& _isoSurfaceColor) {
    if (isoSurfaceColor != _isoSurfaceColor) {
        isoSurfaceColor = _isoSurfaceColor;
        reRender = true;
        frameInfo.frameCount = 0;
    }
}

void VolumetricPathTracingPass::setIsosurfaceType(IsosurfaceType _isosurfaceType) {
    if (isosurfaceType != _isosurfaceType) {
        isosurfaceType = _isosurfaceType;
        setShaderDirty();
        setGridData();
        reRender = true;
        frameInfo.frameCount = 0;
    }
}

void VolumetricPathTracingPass::setSurfaceBrdf(SurfaceBrdf _surfaceBrdf) {
    if (surfaceBrdf != _surfaceBrdf) {
        surfaceBrdf = _surfaceBrdf;
        setShaderDirty();
        reRender = true;
        frameInfo.frameCount = 0;
    }
}

void VolumetricPathTracingPass::setNumIsosurfaceSubdivisions(int _subdivs) {
    if (numIsosurfaceSubdivisions != _subdivs) {
        numIsosurfaceSubdivisions = _subdivs;
        setShaderDirty();
        reRender = true;
        frameInfo.frameCount = 0;
    }
}

void VolumetricPathTracingPass::setFileDialogInstance(ImGuiFileDialog* _fileDialogInstance) {
    this->fileDialogInstance = _fileDialogInstance;
}

void VolumetricPathTracingPass::setDenoiserType(DenoiserType _denoiserType) {
    if (denoiserType != _denoiserType) {
        denoiserType = _denoiserType;
        denoiserChanged = true;
        reRender = true;
    }
}

void VolumetricPathTracingPass::setOutputForegroundMap(bool _shallOutputForegroundMap) {
    if (shallOutputForegroundMap != _shallOutputForegroundMap) {
        shallOutputForegroundMap = _shallOutputForegroundMap;
        denoiseAlpha = _shallOutputForegroundMap;
        setShaderDirty();
#ifdef SUPPORT_OPTIX
        if (denoiserType == DenoiserType::OPTIX) {
            denoiserChanged = true;
        }
#endif
        reRender = true;
        frameInfo.frameCount = 0;
    }
}

void VolumetricPathTracingPass::setSecondaryVolumeDownscalingFactor(uint32_t ds) {
    if (dsSecondaryVolume != ds) {
        dsSecondaryVolume = ds;
        transmittanceVolumeTexture = {};
        if (lastViewportWidth != 0 && lastViewportHeight != 0) {
            checkRecreateFeatureMaps();
        }
    }
}

uint32_t VolumetricPathTracingPass::getSecondaryVolumeDownscalingFactor() {
    return dsSecondaryVolume;
}

size_t VolumetricPathTracingPass::getSecondaryVolumeSizeInBytes() {
    uint32_t width = sgl::uiceil(cloudData->getGridSizeX(), dsSecondaryVolume);
    uint32_t height = sgl::uiceil(cloudData->getGridSizeY(), dsSecondaryVolume);
    uint32_t depth = sgl::uiceil(cloudData->getGridSizeZ(), dsSecondaryVolume);
    return size_t(width) * size_t(height) * size_t(depth) * sizeof(float);
}

void VolumetricPathTracingPass::onHasMoved() {
    frameInfo.frameCount = 0;
}

void VolumetricPathTracingPass::updateVptMode() {
    if (accumulationTimer && !reachedTarget) {
        createNewAccumulationTimer = true;
    }
    if (vptMode == VptMode::ISOSURFACE_RENDERING) {
        useIsosurfaces = true;
    }
    if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING && cloudData && !useSparseGrid) {
        superVoxelGridDecompositionTracking = {};
        superVoxelGridResidualRatioTracking = std::make_shared<SuperVoxelGridResidualRatioTracking>(
                device, cloudData->getGridSizeX(), cloudData->getGridSizeY(),
                cloudData->getGridSizeZ(), cloudData->getDenseDensityField(),
                superVoxelSize, clampToZeroBorder, gridInterpolationType);
        superVoxelGridResidualRatioTracking->setExtinction((cloudExtinctionBase * cloudExtinctionScale).x);
    } else if (vptMode == VptMode::DECOMPOSITION_TRACKING && cloudData && !useSparseGrid) {
        superVoxelGridResidualRatioTracking = {};
        superVoxelGridDecompositionTracking = std::make_shared<SuperVoxelGridDecompositionTracking>(
                device, cloudData->getGridSizeX(), cloudData->getGridSizeY(),
                cloudData->getGridSizeZ(), cloudData->getDenseDensityField(),
                superVoxelSize, clampToZeroBorder, gridInterpolationType);
    } else {
        superVoxelGridResidualRatioTracking = {};
        superVoxelGridDecompositionTracking = {};
    }
}

void VolumetricPathTracingPass::setUseEnvironmentMapFlag(bool useEnvironmentMap) {
    this->useEnvironmentMapImage = useEnvironmentMap;
}

void VolumetricPathTracingPass::setEnvironmentMapIntensityFactor(float intensityFactor) {
    this->environmentMapIntensityFactor = intensityFactor;
}

void VolumetricPathTracingPass::setScatteringAlbedo(glm::vec3 albedo) {
    this->cloudScatteringAlbedo = albedo;
}

void VolumetricPathTracingPass::setExtinctionScale(double extinctionScale){
    this->cloudExtinctionScale = extinctionScale;
}
void VolumetricPathTracingPass::setExtinctionBase(glm::vec3 extinctionBase){
    this->cloudExtinctionBase = extinctionBase;
}

void VolumetricPathTracingPass::setPhaseG(double phaseG){
    this->uniformData.G = phaseG;
}

void VolumetricPathTracingPass::setFeatureMapType(FeatureMapTypeVpt type) {
    this->featureMapType = type;
    if (lastViewportWidth != 0 && lastViewportHeight != 0) {
        checkRecreateFeatureMaps();
    }
    setShaderDirty();
}

void VolumetricPathTracingPass::setUseFeatureMaps(const std::unordered_set<FeatureMapTypeVpt>& _featureMapSet) {
    this->featureMapSet = _featureMapSet;
    if (lastViewportWidth != 0 && lastViewportHeight != 0) {
        checkRecreateFeatureMaps();
    }
    onHasMoved();
    setShaderDirty();
}


void VolumetricPathTracingPass::createEnvironmentMapOctahedralTexture(uint32_t mip_levels) {
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    sgl::vk::ImageSettings imageSettings;
    imageSettings.imageType = VK_IMAGE_TYPE_2D;
    imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL | VK_IMAGE_USAGE_STORAGE_BIT;
    
    // Resolution of 2^mip_level
    imageSettings.width = 1 << mip_levels;
    imageSettings.height = 1 << mip_levels;
    imageSettings.mipLevels = mip_levels;
    imageSettings.format = VK_FORMAT_R32_SFLOAT;

    sgl::vk::ImageSamplerSettings samplerSettings;
    samplerSettings.addressModeU = samplerSettings.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    environmentMapOctahedralTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);

    equalAreaPass->setInputImage(environmentMapTexture);
    equalAreaPass->setOutputImage(environmentMapOctahedralTexture->getImageView());


    VkCommandBuffer commandBuffer = device->beginSingleTimeCommands(0xFFFFFFFF, false);
    renderer->setCustomCommandBuffer(commandBuffer);
    renderer->beginCommandBuffer();
    

    renderer->transitionImageLayout(environmentMapTexture->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    renderer->transitionImageLayout(environmentMapOctahedralTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
    equalAreaPass->render();
    renderer->transitionImageLayout(environmentMapOctahedralTexture->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    
    environmentMapOctahedralTexture->getImage()->generateMipmaps(commandBuffer);

    renderer->endCommandBuffer();
    renderer->resetCustomCommandBuffer();
    device->endSingleTimeCommands(commandBuffer, 0xFFFFFFFF, false);
}

void VolumetricPathTracingPass::loadEnvironmentMapImage(const std::string& filename) {
    if (!sgl::FileUtils::get()->exists(filename)) {
        sgl::Logfile::get()->writeError(
                "Error in VolumetricPathTracingPass::loadEnvironmentMapImage: The file \""
                + filename + "\" does not exist.");
        return;
    }

    bool newEnvMapImageUsesLinearRgb = true;
    sgl::BitmapPtr bitmap;
#ifdef SUPPORT_OPENEXR
    OpenExrImageInfo imageInfo;
#endif
    if (sgl::FileUtils::get()->hasExtension(filename.c_str(), ".png")) {
        bitmap = std::make_shared<sgl::Bitmap>();
        bitmap->fromFile(filename.c_str());
        newEnvMapImageUsesLinearRgb = false; // Assume by default that .png images store sRGB.
    }
#ifdef SUPPORT_OPENEXR
    else if (sgl::FileUtils::get()->hasExtension(filename.c_str(), ".exr")) {
        bool isLoaded = loadOpenExrImageFile(filename, imageInfo);
        if (!isLoaded) {
            sgl::Logfile::get()->writeError(
                    "Error in VolumetricPathTracingPass::loadEnvironmentMapImage: The file \""
                    + filename + "\" couldn't be opened using OpenEXR.");
            return;
        }
    }
#endif
    else {
        sgl::Logfile::get()->writeError(
                "Error in VolumetricPathTracingPass::loadEnvironmentMapImage: The file \""
                + filename + "\" has an unknown file extension.");
        return;
    }

    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    sgl::vk::ImageSettings imageSettings;
    imageSettings.imageType = VK_IMAGE_TYPE_2D;
    imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    sgl::vk::ImageSamplerSettings samplerSettings;
    samplerSettings.addressModeU = samplerSettings.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;

    void* pixelData;
    uint32_t bytesPerPixel;
    uint32_t width;
    uint32_t height;
    if (bitmap) {
        pixelData = bitmap->getPixels();
        bytesPerPixel = bitmap->getBPP() / 8;
        width = uint32_t(bitmap->getWidth());
        height = uint32_t(bitmap->getHeight());
        imageSettings.format = VK_FORMAT_R8G8B8A8_UNORM;
    }
#ifdef SUPPORT_OPENEXR
    else {
        pixelData = imageInfo.pixelData;
        bytesPerPixel = 8; // 4 * half
        width = imageInfo.width;
        height = imageInfo.height;
        imageSettings.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
#endif
    imageSettings.width = width;
    imageSettings.height = height;

    environmentMapTexture = std::make_shared<sgl::vk::Texture>(device, imageSettings, samplerSettings);
    environmentMapTexture->getImage()->uploadData(width * height * bytesPerPixel, pixelData);
    loadedEnvironmentMapFilename = filename;
    isEnvironmentMapLoaded = true;
    frameInfo.frameCount = 0;

    if (envMapImageUsesLinearRgb != newEnvMapImageUsesLinearRgb) {
        envMapImageUsesLinearRgb = newEnvMapImageUsesLinearRgb;
        setShaderDirty();
    }

#ifdef SUPPORT_OPENEXR
    if (!bitmap) {
        delete[] imageInfo.pixelData;
    }
#endif

    createEnvironmentMapOctahedralTexture(12);
}

void VolumetricPathTracingPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> customPreprocessorDefines;
    if (customSeedOffset != 0) {
        customPreprocessorDefines.insert({ "CUSTOM_SEED_OFFSET", std::to_string(customSeedOffset) });
    }
    if (vptMode == VptMode::DELTA_TRACKING) {
        customPreprocessorDefines.insert({ "USE_DELTA_TRACKING", "" });
    } else if (vptMode == VptMode::SPECTRAL_DELTA_TRACKING) {
        customPreprocessorDefines.insert({ "USE_SPECTRAL_DELTA_TRACKING", "" });
        if (sdtCollisionProbability == SpectralDeltaTrackingCollisionProbability::MAX_BASED) {
            customPreprocessorDefines.insert({ "MAX_BASED_PROBABILITY", "" });
        } else if (sdtCollisionProbability == SpectralDeltaTrackingCollisionProbability::AVG_BASED) {
            customPreprocessorDefines.insert({ "AVG_BASED_PROBABILITY", "" });
        } else { // SpectralDeltaTrackingCollisionProbability::PATH_HISTORY_AVG_BASED
            customPreprocessorDefines.insert({ "PATH_HISTORY_AVG_BASED_PROBABILITY", "" });
        }
    } else if (vptMode == VptMode::RATIO_TRACKING) {
        customPreprocessorDefines.insert({ "USE_RATIO_TRACKING", "" });
    } else if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING) {
        customPreprocessorDefines.insert({ "USE_RESIDUAL_RATIO_TRACKING", "" });
    } else if (vptMode == VptMode::DECOMPOSITION_TRACKING) {
        customPreprocessorDefines.insert({ "USE_DECOMPOSITION_TRACKING", "" });
    }else if (vptMode == VptMode::NEXT_EVENT_TRACKING) {
        customPreprocessorDefines.insert({ "USE_NEXT_EVENT_TRACKING", "" });
    } else if (vptMode == VptMode::NEXT_EVENT_TRACKING_SPECTRAL) {
        customPreprocessorDefines.insert({ "USE_NEXT_EVENT_TRACKING_SPECTRAL", "" });
    } else if (vptMode == VptMode::ISOSURFACE_RENDERING) {
        customPreprocessorDefines.insert({ "USE_ISOSURFACE_RENDERING", "" });
        if (useAoDist) {
            customPreprocessorDefines.insert({ "USE_AO_DIST", "" });
        }
    } else if (vptMode == VptMode::RAY_MARCHING_EMISSION_ABSORPTION) {
        customPreprocessorDefines.insert({ "USE_RAY_MARCHING_EMISSION_ABSORPTION", "" });
        if (compositionModel == CompositionModel::ALPHA_BLENDING) {
            customPreprocessorDefines.insert({ "COMPOSITION_MODEL_ALPHA_BLENDING", "" });
        } else if (compositionModel == CompositionModel::AVERAGE) {
            customPreprocessorDefines.insert({ "COMPOSITION_MODEL_AVERAGE", "" });
        } else if (compositionModel == CompositionModel::MAXIMUM_INTENSITY_PROJECTION) {
            customPreprocessorDefines.insert({ "COMPOSITION_MODEL_MAXIMUM_INTENSITY_PROJECTION", "" });
        }
    }
    if (gridInterpolationType == GridInterpolationType::NEAREST) {
        customPreprocessorDefines.insert({ "GRID_INTERPOLATION_NEAREST", "" });
    } else if (gridInterpolationType == GridInterpolationType::STOCHASTIC) {
        customPreprocessorDefines.insert({ "GRID_INTERPOLATION_STOCHASTIC", "" });
    } else if (gridInterpolationType == GridInterpolationType::TRILINEAR) {
        customPreprocessorDefines.insert({ "GRID_INTERPOLATION_TRILINEAR", "" });
    }
    if (useSparseGrid) {
        customPreprocessorDefines.insert({ "USE_NANOVDB", "" });
    }
    if (useEmission && (emissionFieldTexture || emissionNanoVdbBuffer)) {
        customPreprocessorDefines.insert({ "USE_EMISSION", "" });
    }

    if (firstXTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_POSITION_MAP", ""));
    }
    if (firstWTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_FIRST_W_MAP", ""));
    }
    if (normalTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_NORMAL_MAP", ""));
    }
    if (cloudOnlyTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_CLOUDONLY_MAP", ""));
    }
    if (depthTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_DEPTH_MAP", ""));
    }
    if (densityTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_DENSITY_MAP", ""));
    }
    if (backgroundTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_BACKGROUND_MAP", ""));
    }
    if (reprojUVTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_REPROJ_UV_MAP", ""));
    }
    if (depthBlendedTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_DEPTH_BLENDED_MAP", ""));
    }
    if (flowTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_FLOW_MAP", ""));
    }
    if (depthNablaTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_DEPTH_NABLA_MAP", ""));
    }
    if (depthFwidthTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_DEPTH_FWIDTH_MAP", ""));
    }
    if (transmittanceVolumeTexture) {
        customPreprocessorDefines.insert(std::make_pair("WRITE_TRANSMITTANCE_VOLUME", ""));
    }
    if (denoiser && !denoiser->getWantsAccumulatedInput()) {
        customPreprocessorDefines.insert(std::make_pair("DISABLE_ACCUMULATION", ""));
    }

    if (shallOutputForegroundMap) {
        customPreprocessorDefines.insert(std::make_pair("OUTPUT_FOREGROUND_MAP", ""));
    }

    if (blitPrimaryRayMomentTexturePass->getMomentType() != BlitMomentTexturePass::MomentType::NONE) {
        customPreprocessorDefines.insert({ "COMPUTE_PRIMARY_RAY_ABSORPTION_MOMENTS", "" });
        customPreprocessorDefines.insert(
                { "NUM_PRIMARY_RAY_ABSORPTION_MOMENTS",
                  std::to_string(blitPrimaryRayMomentTexturePass->getNumMoments()) });
        if (blitPrimaryRayMomentTexturePass->getMomentType() == BlitMomentTexturePass::MomentType::POWER) {
            customPreprocessorDefines.insert({ "USE_POWER_MOMENTS_PRIMARY_RAY", "" });
        }
    }
    if (blitScatterRayMomentTexturePass->getMomentType() != BlitMomentTexturePass::MomentType::NONE) {
        customPreprocessorDefines.insert({ "COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS", "" });
        customPreprocessorDefines.insert(
                { "NUM_SCATTER_RAY_ABSORPTION_MOMENTS",
                  std::to_string(blitScatterRayMomentTexturePass->getNumMoments()) });
        if (blitScatterRayMomentTexturePass->getMomentType() == BlitMomentTexturePass::MomentType::POWER) {
            customPreprocessorDefines.insert({ "USE_POWER_MOMENTS_SCATTER_RAY", "" });
        }
    }
    if (useEnvironmentMapImage && isEnvironmentMapLoaded) {
        customPreprocessorDefines.insert({ "USE_ENVIRONMENT_MAP_IMAGE", "" });
    }
    if (uniformData.useLinearRGB) {
        customPreprocessorDefines.insert({ "USE_LINEAR_RGB", "" });
    }
    if (envMapImageUsesLinearRgb) {
        customPreprocessorDefines.insert({ "ENV_MAP_IMAGE_USES_LINEAR_RGB", "" });
    }
    if (flipYZCoordinates) {
        customPreprocessorDefines.insert({ "FLIP_YZ", "" });
    }

    customPreprocessorDefines.insert({ "LOCAL_SIZE_X", std::to_string(blockSize2D.x) });
    customPreprocessorDefines.insert({ "LOCAL_SIZE_Y", std::to_string(blockSize2D.y) });
    auto* tfWindow = cloudData->getTransferFunctionWindow();
    bool useTransferFunction = tfWindow && tfWindow->getShowWindow();
    if (useTransferFunction) {
        customPreprocessorDefines.insert({ "USE_TRANSFER_FUNCTION", "" });
    }
    if (useTransferFunctionCached != useTransferFunction) {
        useTransferFunctionCached = useTransferFunction;
        frameInfo.frameCount = 0;
    }

    if (useIsosurfaces) {
        customPreprocessorDefines.insert({ "USE_ISOSURFACES", "" });
        if (vptMode != VptMode::ISOSURFACE_RENDERING) {
            customPreprocessorDefines.insert({ "NUM_ISOSURFACE_SUBDIVISIONS", std::to_string(numIsosurfaceSubdivisions) });
        }
        if (surfaceBrdf == SurfaceBrdf::LAMBERTIAN) {
            customPreprocessorDefines.insert({ "SURFACE_BRDF_LAMBERTIAN", "" });
        } else if (surfaceBrdf == SurfaceBrdf::BLINN_PHONG) {
            customPreprocessorDefines.insert({ "SURFACE_BRDF_BLINN_PHONG", "" });
        }
        if (isosurfaceType == IsosurfaceType::DENSITY) {
            customPreprocessorDefines.insert({ "ISOSURFACE_TYPE_DENSITY", "" });
        } else if (isosurfaceType == IsosurfaceType::GRADIENT) {
            customPreprocessorDefines.insert({ "ISOSURFACE_TYPE_GRADIENT", "" });
        }
        if (vptMode == VptMode::NEXT_EVENT_TRACKING || vptMode == VptMode::NEXT_EVENT_TRACKING_SPECTRAL) {
            customPreprocessorDefines.insert({ "USE_ISOSURFACE_NEE", "" });
        }
    }

    shaderStages = sgl::vk::ShaderManager->getShaderStages({"Clouds.Compute"}, customPreprocessorDefines);
}

void VolumetricPathTracingPass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticImageView(resultImageView, "resultImage");
    if (useSparseGrid) {
        computeData->setStaticBuffer(nanoVdbBuffer, "NanoVdbBuffer");
        if (useEmission && emissionNanoVdbBuffer){
            computeData->setStaticBuffer(emissionNanoVdbBuffer, "EmissionNanoVdbBuffer");
        }
    } else {
        computeData->setStaticTexture(densityFieldTexture, "gridImage");
        if (useEmission && emissionFieldTexture){
            computeData->setStaticTexture(emissionFieldTexture, "emissionImage");
        }
        if (useIsosurfaces && isosurfaceType == IsosurfaceType::GRADIENT) {
            computeData->setStaticTexture(densityGradientFieldTexture, "gradientImage");
        }
        if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING) {
            computeData->setStaticTexture(
                    superVoxelGridResidualRatioTracking->getSuperVoxelGridTexture(),
                    "superVoxelGridImage");
            computeData->setStaticTexture(
                    superVoxelGridResidualRatioTracking->getSuperVoxelGridOccupancyTexture(),
                    "superVoxelGridOccupancyImage");
        } else if (vptMode == VptMode::DECOMPOSITION_TRACKING) {
            computeData->setStaticTexture(
                    superVoxelGridDecompositionTracking->getSuperVoxelGridTexture(),
                    "superVoxelGridImage");
            computeData->setStaticTexture(
                    superVoxelGridDecompositionTracking->getSuperVoxelGridOccupancyTexture(),
                    "superVoxelGridOccupancyImage");
        }
    }
    computeData->setStaticBuffer(uniformBuffer, "Parameters");
    computeData->setStaticBuffer(frameInfoBuffer, "FrameInfo");
    computeData->setStaticImageView(accImageTexture->getImageView(), "accImage");

    if (firstXTexture) {
        computeData->setStaticImageView(firstXTexture->getImageView(), "firstX");
    }
    if (firstWTexture) {
        computeData->setStaticImageView(firstWTexture->getImageView(), "firstW");
    }
    if (normalTexture) {
        computeData->setStaticImageView(normalTexture->getImageView(), "normalImage");
    }
    if (cloudOnlyTexture) {
        computeData->setStaticImageView(cloudOnlyTexture->getImageView(), "cloudOnlyImage");
    }
    if (depthTexture) {
        computeData->setStaticImageView(depthTexture->getImageView(), "depthImage");
    }
    if (densityTexture) {
        computeData->setStaticImageView(densityTexture->getImageView(), "densityImage");
    }
    if (backgroundTexture) {
        computeData->setStaticImageView(backgroundTexture->getImageView(), "backgroundImage");
    }
    if (reprojUVTexture) {
        computeData->setStaticImageView(reprojUVTexture->getImageView(), "reprojUVImage");
    }
    if (depthBlendedTexture) {
        computeData->setStaticImageView(depthBlendedTexture->getImageView(), "depthBlendedImage");
    }
    if (flowTexture) {
        computeData->setStaticImageView(flowTexture->getImageView(), "flowImage");
    }
    if (depthNablaTexture) {
        computeData->setStaticImageView(depthNablaTexture->getImageView(), "depthNablaImage");
    }
    if (depthFwidthTexture) {
        computeData->setStaticImageView(depthFwidthTexture->getImageView(), "depthFwidthImage");
    }
    if (transmittanceVolumeTexture) {
        computeData->setStaticImageView(transmittanceVolumeTexture->getImageView(), "transmittanceVolumeImage");
    }

    if (useEnvironmentMapImage && isEnvironmentMapLoaded) {
        computeData->setStaticTexture(environmentMapTexture, "environmentMapTexture");
        computeData->setStaticTexture(environmentMapOctahedralTexture, "environmentMapOctahedralTexture");
    }

    if (blitPrimaryRayMomentTexturePass->getMomentType() != BlitMomentTexturePass::MomentType::NONE) {
        computeData->setStaticImageView(
                blitPrimaryRayMomentTexturePass->getMomentTexture()->getImageView(),
                "primaryRayAbsorptionMomentsImage");
    }
    if (blitScatterRayMomentTexturePass->getMomentType() != BlitMomentTexturePass::MomentType::NONE) {
        computeData->setStaticImageView(
                blitScatterRayMomentTexturePass->getMomentTexture()->getImageView(),
                "scatterRayAbsorptionMomentsImage");
    }
    computeData->setStaticBuffer(momentUniformDataBuffer, "MomentUniformData");

    auto* tfWindow = cloudData->getTransferFunctionWindow();
    if (tfWindow && tfWindow->getShowWindow()) {
        computeData->setStaticTexture(tfWindow->getTransferFunctionMapTextureVulkan(), "transferFunctionTexture");
    }
}

std::string VolumetricPathTracingPass::getCurrentEventName() {
    return std::string() + VPT_MODE_NAMES[int(vptMode)] + " " + std::to_string(targetNumSamples) + "spp";
}

void VolumetricPathTracingPass::checkRecreateDenoiser() {
    if (denoiserChanged) {
        createDenoiser();
        denoiserChanged = false;
    }
}

void VolumetricPathTracingPass::_render() {
    checkRecreateDenoiser();

    std::string eventName = getCurrentEventName();
    if (createNewAccumulationTimer) {
        accumulationTimer = {};
        accumulationTimer = std::make_shared<sgl::vk::Timer>(renderer);
        denoiseTimer = std::make_shared<sgl::vk::Timer>(renderer);
        createNewAccumulationTimer = false;
    }

    if (!reachedTarget) {
        if (int(frameInfo.frameCount) > targetNumSamples) {
            frameInfo.frameCount = 0;
        }
        if (int(frameInfo.frameCount) < targetNumSamples) {
            reRender = true;
        }
        if (int(frameInfo.frameCount) == targetNumSamples) {
            reachedTarget = true;
            accumulationTimer->finishGPU();
            accumulationTimer->printTimeMS(eventName);
            denoiseTimer->finishGPU();
            if (useDenoiser && denoiser) {
                denoiseTimer->printTimeMS("denoise");
                accumulationTimer->printTimeMS("denoise");
            }

            timerStopped = true;
        }
    }

    if (!reachedTarget) {
        accumulationTimer->startGPU(eventName);
    }

    if (frameInfo.frameCount == 0 || (!changedDenoiserSettings && !timerStopped)) {
        uniformData.inverseViewProjMatrix = glm::inverse(
                (*camera)->getProjectionMatrix() * (*camera)->getViewMatrix());

        uniformData.previousViewProjMatrix = previousViewProjMatrix;
        if (previousViewProjMatrix[3][3] == 0){
            // No previous view projection matrix.
            uniformData.previousViewProjMatrix = (*camera)->getProjectionMatrix() * (*camera)->getViewMatrix();
        }
        uniformData.inverseTransposedViewMatrix = glm::transpose(glm::inverse((*camera)->getViewMatrix()));
        uniformData.farDistance = (*camera)->getFarClipDistance();
        uniformData.boxMin = cloudData->getWorldSpaceBoxMin(useSparseGrid);
        uniformData.boxMax = cloudData->getWorldSpaceBoxMax(useSparseGrid);
        if (emissionData){
            uniformData.emissionBoxMin = emissionData->getWorldSpaceBoxMin(useSparseGrid);
            uniformData.emissionBoxMax = emissionData->getWorldSpaceBoxMax(useSparseGrid);
        }
        if (flipYZCoordinates){
            uniformData.boxMin.y = cloudData->getWorldSpaceBoxMin(useSparseGrid).z;
            uniformData.boxMin.z = cloudData->getWorldSpaceBoxMin(useSparseGrid).y;
            uniformData.boxMax.y = cloudData->getWorldSpaceBoxMax(useSparseGrid).z;
            uniformData.boxMax.z = cloudData->getWorldSpaceBoxMax(useSparseGrid).y;
            if (emissionData){
                uniformData.emissionBoxMin.y = emissionData->getWorldSpaceBoxMin(useSparseGrid).z;
                uniformData.emissionBoxMin.z = emissionData->getWorldSpaceBoxMin(useSparseGrid).y;
                uniformData.emissionBoxMax.y = emissionData->getWorldSpaceBoxMax(useSparseGrid).z;
                uniformData.emissionBoxMax.z = emissionData->getWorldSpaceBoxMax(useSparseGrid).y;
            }
        }
        if (useSparseGrid) {
            uniformData.gridMin = cloudData->getWorldSpaceSparseGridMin();
            uniformData.gridMax = cloudData->getWorldSpaceSparseGridMax();
        } else {
            uniformData.gridMin = cloudData->getWorldSpaceDenseGridMin();
            uniformData.gridMax = cloudData->getWorldSpaceDenseGridMax();
        }
        if (!useSparseGrid){
            uniformData.gridMin = glm::vec3(0,0,0);
            uniformData.gridMax = glm::vec3(1,1,1);
        }

        if (!useSparseGrid) {
            auto densityField = cloudData->getDenseDensityField();
            uniformData.voxelValueMin = densityField->getMinValue();
            uniformData.voxelValueMax = densityField->getMaxValue();
        } else {
            uniformData.voxelValueMin = 0.0f;
            uniformData.voxelValueMax = 1.0f;
        }
        if (uniformData.voxelValueMin == uniformData.voxelValueMax) {
            uniformData.voxelValueMin = 0.0f;
        }

        uniformData.emissionCap = emissionCap;
        uniformData.emissionStrength = emissionStrength;
        uniformData.extinction = cloudExtinctionBase * cloudExtinctionScale;
        uniformData.scatteringAlbedo = cloudScatteringAlbedo;
        uniformData.sunDirection = sunlightDirection;
        uniformData.sunIntensity = sunlightIntensity * sunlightColor;
        uniformData.environmentMapIntensityFactor = environmentMapIntensityFactor;
        uniformData.numFeatureMapSamplesPerFrame = numFeatureMapSamplesPerFrame;
        if (useSparseGrid) {
            if (cloudData->getGridSizeX() >= 8 && cloudData->getGridSizeY() >= 8 && cloudData->getGridSizeZ() >= 8) {
                uniformData.superVoxelSize = glm::ivec3(8);
            } else {
                uniformData.superVoxelSize = glm::ivec3(1);
            }
        } else if (superVoxelGridResidualRatioTracking) {
            uniformData.superVoxelSize = superVoxelGridResidualRatioTracking->getSuperVoxelSize();
            uniformData.superVoxelGridSize = superVoxelGridResidualRatioTracking->getSuperVoxelGridSize();
        } else if (superVoxelGridDecompositionTracking) {
            uniformData.superVoxelSize = superVoxelGridDecompositionTracking->getSuperVoxelSize();
            uniformData.superVoxelGridSize = superVoxelGridDecompositionTracking->getSuperVoxelGridSize();
        }

        if (!useSparseGrid) {
            auto settings = densityFieldTexture->getImage()->getImageSettings();
            uniformData.voxelTexelSize =
                    glm::vec3(1.0f) / glm::vec3(settings.width - 1, settings.height - 1, settings.depth - 1);
        } else {
            uniformData.voxelTexelSize = glm::vec3(1.0f) / glm::vec3(
                    cloudData->getGridSizeX() - 1, cloudData->getGridSizeY() - 1, cloudData->getGridSizeZ() - 1);
        }

        uniformData.isoSurfaceColor = isoSurfaceColor;
        uniformData.isoValue = isoValue;
        uniformData.isoStepWidth = isoStepWidth;
        uniformData.maxAoDist = maxAoDist;
        uniformData.numAoSamples = numAoSamples;
        uniformBuffer->updateData(
                sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());

        if (useGlobalFrameNumber) {
            frameInfo.globalFrameNumber = globalFrameNumber;
        } else {
            frameInfo.globalFrameNumber = frameInfo.frameCount;
        }
        frameInfoBuffer->updateData(
                sizeof(FrameInfo), &frameInfo, renderer->getVkCommandBuffer());
        frameInfo.frameCount++;
        globalFrameNumber++;

        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        renderer->transitionImageLayout(resultImageView->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        if (!useSparseGrid) {
            renderer->transitionImageLayout(
                    densityFieldTexture->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        renderer->transitionImageLayout(accImageTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        if (firstXTexture) {
            renderer->transitionImageLayout(firstXTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (firstWTexture) {
            renderer->transitionImageLayout(firstWTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (normalTexture) {
            renderer->transitionImageLayout(normalTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (cloudOnlyTexture) {
            renderer->transitionImageLayout(cloudOnlyTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (backgroundTexture) {
            renderer->transitionImageLayout(backgroundTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (reprojUVTexture) {
            renderer->transitionImageLayout(reprojUVTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (depthTexture) {
            renderer->transitionImageLayout(depthTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (densityTexture) {
            renderer->transitionImageLayout(densityTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (depthBlendedTexture) {
            renderer->transitionImageLayout(depthBlendedTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (flowTexture) {
            renderer->transitionImageLayout(flowTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (depthNablaTexture) {
            renderer->transitionImageLayout(depthNablaTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (depthFwidthTexture) {
            renderer->transitionImageLayout(depthFwidthTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        }
        if (transmittanceVolumeTexture) {
            renderer->insertImageMemoryBarrier(
                    transmittanceVolumeTexture->getImage(),
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_NONE_KHR, VK_ACCESS_TRANSFER_WRITE_BIT);
            transmittanceVolumeTexture->getImageView()->clearColor(glm::vec4(0.0f), renderer->getVkCommandBuffer());
            renderer->insertImageMemoryBarrier(
                    transmittanceVolumeTexture->getImage(),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        //renderer->transitionImageLayout(
        //        blitPrimaryRayMomentTexturePass->getMomentTexture()->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        //renderer->transitionImageLayout(
        //        blitScatterRayMomentTexturePass->getMomentTexture()->getImage(), VK_IMAGE_LAYOUT_GENERAL);
        if (blitPrimaryRayMomentTexturePass->getMomentType() != BlitMomentTexturePass::MomentType::NONE) {
            renderer->transitionImageLayout(
                    blitPrimaryRayMomentTexturePass->getMomentTexture()->getImage(),
                    VK_IMAGE_LAYOUT_GENERAL);
        }
        if (blitScatterRayMomentTexturePass->getMomentType() != BlitMomentTexturePass::MomentType::NONE) {
            renderer->transitionImageLayout(
                    blitScatterRayMomentTexturePass->getMomentTexture()->getImage(),
                    VK_IMAGE_LAYOUT_GENERAL);
        }

        auto& imageSettings = resultImageView->getImage()->getImageSettings();
        renderer->dispatch(
                computeData,
                sgl::iceil(int(imageSettings.width), blockSize2D.x),
                sgl::iceil(int(imageSettings.height), blockSize2D.y),
                1);
    }
    changedDenoiserSettings = false;
    timerStopped = false;

    if (featureMapType == FeatureMapTypeVpt::RESULT) {
        if (useDenoiser && denoiser && denoiser->getIsEnabled() && !isIntermediatePass) {
            if (!reachedTarget){
                denoiseTimer->startGPU("denoise");
                accumulationTimer->startGPU("denoise");
            }
            denoiser->denoise();
            if (!reachedTarget){
                accumulationTimer->endGPU("denoise");
                denoiseTimer->endGPU("denoise");
            }
            renderer->transitionImageLayout(
                    denoisedImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            renderer->transitionImageLayout(
                    sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            denoisedImageView->getImage()->blit(
                    sceneImageView->getImage(), renderer->getVkCommandBuffer());
        } else {
            /*renderer->transitionImageLayout(
                    resultImageView->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            blitResultRenderPass->render();*/
            renderer->transitionImageLayout(
                    resultImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            renderer->transitionImageLayout(
                    sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            resultImageView->getImage()->blit(
                    sceneImageView->getImage(), renderer->getVkCommandBuffer());
        }
        if (cameraPosesSet) {
            cameraPoseLinePass->setMvpMatrix((*camera)->getViewProjMatrix());
            cameraPoseLinePass->render();
        }
    } else if (featureMapType == FeatureMapTypeVpt::FIRST_X) {
        renderer->transitionImageLayout(firstXTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        firstXTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::FIRST_W) {
        renderer->transitionImageLayout(firstWTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        firstWTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::NORMAL) {
        renderer->transitionImageLayout(normalTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        normalTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::CLOUD_ONLY) {
        renderer->transitionImageLayout(cloudOnlyTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        cloudOnlyTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::DEPTH) {
        renderer->transitionImageLayout(depthTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        depthTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::DENSITY) {
        renderer->transitionImageLayout(densityTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        densityTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::BACKGROUND) {
        renderer->transitionImageLayout(backgroundTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        backgroundTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::REPROJ_UV) {
        renderer->transitionImageLayout(reprojUVTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        reprojUVTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::DEPTH_BLENDED) {
        renderer->transitionImageLayout(depthBlendedTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        depthBlendedTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::FLOW) {
        renderer->transitionImageLayout(flowTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        flowTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::DEPTH_NABLA) {
        renderer->transitionImageLayout(depthNablaTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        depthNablaTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::DEPTH_FWIDTH) {
        renderer->transitionImageLayout(depthFwidthTexture->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        depthFwidthTexture->getImage()->blit(sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::TRANSMITTANCE_VOLUME) {
        sgl::Logfile::get()->writeWarning(
                "Warning: Transmittance volume is a 3D feature map and cannot be displayed.", true);
        renderer->transitionImageLayout(
                resultImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        renderer->transitionImageLayout(
                sceneImageView->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        resultImageView->getImage()->blit(
                sceneImageView->getImage(), renderer->getVkCommandBuffer());
    } else if (featureMapType == FeatureMapTypeVpt::PRIMARY_RAY_ABSORPTION_MOMENTS) {
        blitPrimaryRayMomentTexturePass->renderOptional();
    } else if (featureMapType == FeatureMapTypeVpt::SCATTER_RAY_ABSORPTION_MOMENTS) {
        blitScatterRayMomentTexturePass->renderOptional();
    }

    if (!reachedTarget) {
        accumulationTimer->endGPU(eventName);
    }
    this->setPreviousViewProjMatrix((*camera)->getProjectionMatrix() * (*camera)->getViewMatrix());

    if (cloudData->getNextCloudDataFrame() != nullptr){
        this->setCloudData(cloudData->getNextCloudDataFrame());
    }
    if (emissionData && emissionData->getNextCloudDataFrame() != nullptr){
        this->setEmissionData(emissionData->getNextCloudDataFrame());
    }
}

sgl::vk::TexturePtr VolumetricPathTracingPass::getFeatureMapTexture(FeatureMapTypeVpt type){
    if (type == FeatureMapTypeVpt::RESULT) {
        return accImageTexture;
    } else if (type == FeatureMapTypeVpt::FIRST_X) {
        return firstXTexture;
    } else if (type == FeatureMapTypeVpt::FIRST_W) {
        return firstWTexture;
    } else if (type == FeatureMapTypeVpt::NORMAL) {
        return normalTexture;
    //} else if (type == FeatureMapTypeVpt::PRIMARY_RAY_ABSORPTION_MOMENTS) {
    //    return nullptr;
    //} else if (type == FeatureMapTypeVpt::SCATTER_RAY_ABSORPTION_MOMENTS) {
    //    return nullptr;
    } else if (type == FeatureMapTypeVpt::CLOUD_ONLY) {
        return cloudOnlyTexture;
    } else if (type == FeatureMapTypeVpt::DEPTH) {
        return depthTexture;
    } else if (type == FeatureMapTypeVpt::DENSITY) {
        return densityTexture;
    } else if (type == FeatureMapTypeVpt::BACKGROUND) {
        return backgroundTexture;
    } else if (type == FeatureMapTypeVpt::REPROJ_UV) {
        return reprojUVTexture;
    } else if (type == FeatureMapTypeVpt::DEPTH_BLENDED) {
        return depthBlendedTexture;
    } else if (type == FeatureMapTypeVpt::FLOW) {
        return flowTexture;
    } else if (type == FeatureMapTypeVpt::DEPTH_NABLA) {
        return depthNablaTexture;
    } else if (type == FeatureMapTypeVpt::DEPTH_FWIDTH) {
        return depthFwidthTexture;
    } else if (type == FeatureMapTypeVpt::TRANSMITTANCE_VOLUME) {
        return transmittanceVolumeTexture;
    }
    return nullptr;
}


bool VolumetricPathTracingPass::renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) {
    bool optionChanged = false;

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseEnvironmentMapImage", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathNameString(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilterString(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            // Is this line data set or a volume data file for the scattering line tracer?
            std::string currentPath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            filename += selection.table[0].fileName;
            IGFD_Selection_DestroyContent(&selection);

            environmentMapFilenameGui = filename;
            loadEnvironmentMapImage(environmentMapFilenameGui);
            setShaderDirty();
            reRender = true;
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    if (propertyEditor.beginNode("VPT Renderer")) {
        std::string numSamplesText = "#Samples: " + std::to_string(frameInfo.frameCount) + "###numSamplesText";
        propertyEditor.addCustomWidgets(numSamplesText);
        if (ImGui::Button(" = ")) {
            createNewAccumulationTimer = true;
            reachedTarget = false;
            reRender = true;

            if (int(frameInfo.frameCount) >= targetNumSamples) {
                frameInfo.frameCount = 0;
            }
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(sgl::ImGuiWrapper::get()->getScaleDependentSize(220.0f));
        ImGui::InputInt("##targetNumSamples", &targetNumSamples);

        propertyEditor.addCustomWidgets("#Feature Samples");
        ImGui::InputInt("##samplesPerFrame", &numFeatureMapSamplesPerFrame);

        if (propertyEditor.addSliderFloat("Extinction Scale", &cloudExtinctionScale, 1.0f, 2048.0f)) {
            optionChanged = true;
        }
        if (propertyEditor.addSliderFloat3("Extinction Base", &cloudExtinctionBase.x, 0.01f, 1.0f)) {
            optionChanged = true;
        }
        if (propertyEditor.addColorEdit3("Scattering Albedo", &cloudScatteringAlbedo.x, ImGuiColorEditFlags_Float)) {
            optionChanged = true;
        }
        if (propertyEditor.addSliderFloat("G", &uniformData.G, -1.0f, 1.0f)) {
            optionChanged = true;
        }
        if (propertyEditor.addCombo(
                "Feature Map", (int*)&featureMapType, VPT_FEATURE_MAP_NAMES,
                IM_ARRAYSIZE(VPT_FEATURE_MAP_NAMES))) {
            checkRecreateFeatureMaps();
            setShaderDirty();
            optionChanged = true;
            //blitPrimaryRayMomentTexturePass->setVisualizeMomentTexture(
            //        featureMapType == FeatureMapTypeVpt::PRIMARY_RAY_ABSORPTION_MOMENTS);
            //blitScatterRayMomentTexturePass->setVisualizeMomentTexture(
            //        featureMapType == FeatureMapTypeVpt::SCATTER_RAY_ABSORPTION_MOMENTS);
        }
        if (propertyEditor.addCombo(
                "VPT Mode", (int*)&vptMode, VPT_MODE_NAMES,
                IM_ARRAYSIZE(VPT_MODE_NAMES))) {
            optionChanged = true;
            if (vptMode == VptMode::ISOSURFACE_RENDERING && gridInterpolationType != GridInterpolationType::TRILINEAR) {
                gridInterpolationType = GridInterpolationType::TRILINEAR;
                updateGridSampler();
            }
            updateVptMode();
            setShaderDirty();
            setDataDirty();
        }

        if (vptMode == VptMode::SPECTRAL_DELTA_TRACKING) {
            if (propertyEditor.addCombo(
                    "Collision Probability", (int*)&sdtCollisionProbability,
                    SPECTRAL_DELTA_TRACKING_COLLISION_PROBABILITY_NAMES,
                    IM_ARRAYSIZE(SPECTRAL_DELTA_TRACKING_COLLISION_PROBABILITY_NAMES))) {
                optionChanged = true;
                setShaderDirty();
            }
        }

        if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING || vptMode == VptMode::DECOMPOSITION_TRACKING) {
            if (propertyEditor.addSliderInt("Super Voxel Size", &superVoxelSize, 1, 64)) {
                optionChanged = true;
                updateVptMode();
                setShaderDirty();
                setDataDirty();
            }
        }

        if (vptMode == VptMode::RAY_MARCHING_EMISSION_ABSORPTION) {
            if (propertyEditor.addCombo(
                    "Composition Model", (int*)&compositionModel,
                    COMPOSITION_MODEL_NAMES, IM_ARRAYSIZE(COMPOSITION_MODEL_NAMES))) {
                optionChanged = true;
                setShaderDirty();
            }
        }

        if (propertyEditor.addCheckbox("Use Sparse Grid", &useSparseGrid)) {
            optionChanged = true;
            setGridData();
            updateVptMode();
            setShaderDirty();
            setDataDirty();
        }
        if (propertyEditor.addCheckbox("Flip YZ", &flipYZCoordinates)) {
            optionChanged = true;
            setGridData();
            updateVptMode();
            setShaderDirty();
            setDataDirty();
        }

        if (!useIsosurfaces && propertyEditor.addCombo(
                "Grid Interpolation", (int*)&gridInterpolationType,
                GRID_INTERPOLATION_TYPE_NAMES, IM_ARRAYSIZE(GRID_INTERPOLATION_TYPE_NAMES))) {
            optionChanged = true;
            if (vptMode == VptMode::RESIDUAL_RATIO_TRACKING || vptMode == VptMode::DECOMPOSITION_TRACKING) {
                updateVptMode();
            }
            updateGridSampler();
            setShaderDirty();
        }



        if (propertyEditor.addCheckbox(
                "Use Env. Map Image", &useEnvironmentMapImage)) {
            setShaderDirty();
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (useEnvironmentMapImage){
            propertyEditor.addInputAction("Environment Map", &environmentMapFilenameGui);
            if (propertyEditor.addButton("", "Load")) {
                loadEnvironmentMapImage(environmentMapFilenameGui);
                setShaderDirty();
                reRender = true;
            }
            ImGui::SameLine();
            if (ImGui::Button("Open from Disk...")) {
                IGFD_OpenModal(
                        fileDialogInstance,
                        "ChooseEnvironmentMapImage", "Choose an Environment Map Image",
                        ".*,.png,.exr",
                        sgl::AppSettings::get()->getDataDirectory().c_str(),
                        "", 1, nullptr,
                        ImGuiFileDialogFlags_None);
            }
            if (useEnvironmentMapImage && propertyEditor.addSliderFloat(
                    "Env. Map Intensity", &environmentMapIntensityFactor, 0.0f, 5.0f)) {
                reRender = true;
                frameInfo.frameCount = 0;
            }

            if (useEnvironmentMapImage && propertyEditor.addCheckbox(
                    "Env. Map Linear RGB", &envMapImageUsesLinearRgb)) {
                reRender = true;
                frameInfo.frameCount = 0;
                setShaderDirty();
            }
        } else {
            if (propertyEditor.addColorEdit3("Sunlight Color", &sunlightColor.x)) {
                optionChanged = true;
            }
            if (propertyEditor.addSliderFloat("Sunlight Intensity", &sunlightIntensity, 0.0f, 10.0f)) {
                optionChanged = true;
            }
            if (propertyEditor.addSliderFloat3("Sunlight Direction", &sunlightDirection.x, 0.0f, 1.0f)) {
                optionChanged = true;
            }
        }



        if (propertyEditor.addCheckbox("Use Emission", &useEmission)) {
            optionChanged = true;
            setGridData();
            updateVptMode();
            setShaderDirty();
            setDataDirty();
        }
        if (propertyEditor.addSliderFloat(
                "Emission Cap", &emissionCap, 0.0f, 1.0f)) {
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (propertyEditor.addSliderFloat(
                "Emission Strength", &emissionStrength, 0.0f, 20.0f)) {
            reRender = true;
            frameInfo.frameCount = 0;
        }
        propertyEditor.addInputAction("Emission Grid", &emissionGridFilenameGui);
        if (propertyEditor.addCheckbox("Load", &useEmission)) {
            CloudDataPtr emissionCloudData(new CloudData);
            bool dataLoaded = emissionCloudData->loadFromFile(emissionGridFilenameGui);
            if (dataLoaded){
                setEmissionData(emissionCloudData);
            }
            setShaderDirty();
            reRender = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Open from Disk...")) {
            IGFD_OpenModal(
                    fileDialogInstance,
                    "ChooseEmissionGrid", "Choose an Emission Grid",
                    ".*",
                    sgl::AppSettings::get()->getDataDirectory().c_str(),
                    "", 1, nullptr,
                    ImGuiFileDialogFlags_None);
        }

        bool shallRecreateMomentTextureA = false;
        bool momentTypeChangedA = false;
        bool shallRecreateMomentTextureB = false;
        bool momentTypeChangedB = false;
        optionChanged =
                blitPrimaryRayMomentTexturePass->renderGuiPropertyEditorNodes(
                        propertyEditor, shallRecreateMomentTextureA, momentTypeChangedA) || optionChanged;
        optionChanged =
                blitScatterRayMomentTexturePass->renderGuiPropertyEditorNodes(
                        propertyEditor, shallRecreateMomentTextureB, momentTypeChangedB) || optionChanged;
        if (shallRecreateMomentTextureA || shallRecreateMomentTextureB) {
            setShaderDirty();
            setDataDirty();
        }
        if (momentTypeChangedA || momentTypeChangedB) {
            setShaderDirty();
        }

        if (vptMode != VptMode::ISOSURFACE_RENDERING && propertyEditor.addCheckbox(
                "Use Isosurfaces", &useIsosurfaces)) {
            if (gridInterpolationType != GridInterpolationType::TRILINEAR) {
                gridInterpolationType = GridInterpolationType::TRILINEAR;
                updateGridSampler();
            }
            setShaderDirty();
            if (isosurfaceType == IsosurfaceType::GRADIENT) {
                setGridData();
            }
            reRender = true;
            frameInfo.frameCount = 0;
        }
        float isoValMin = isosurfaceType == IsosurfaceType::DENSITY ? 0.0f : minGradientVal;
        float isoValMax = isosurfaceType == IsosurfaceType::DENSITY ? 1.0f : maxGradientVal;
        if (useIsosurfaces && propertyEditor.addSliderFloat("Iso Value", &isoValue, isoValMin, isoValMax)) {
            setShaderDirty();
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (useIsosurfaces && propertyEditor.addColorEdit3("Isosurface Color", &isoSurfaceColor.x)) {
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (useIsosurfaces && vptMode != VptMode::ISOSURFACE_RENDERING && propertyEditor.addSliderInt(
                "#Isosurface Subdivisions", &numIsosurfaceSubdivisions, 1, 8)) {
            setShaderDirty();
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (vptMode == VptMode::ISOSURFACE_RENDERING && propertyEditor.addSliderFloat(
                "Step Width", &isoStepWidth, 0.1f, 1.0f)) {
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (vptMode == VptMode::ISOSURFACE_RENDERING && propertyEditor.addSliderFloat(
                "AO Dist.", &maxAoDist, 0.01f, 0.1f)) {
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (vptMode == VptMode::ISOSURFACE_RENDERING && propertyEditor.addSliderInt(
                "#AO Samples", &numAoSamples, 1, 16)) {
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (vptMode == VptMode::ISOSURFACE_RENDERING && propertyEditor.addCheckbox("Use AO Dist.", &useAoDist)) {
            setShaderDirty();
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (useIsosurfaces && propertyEditor.addCombo(
                "Isosurface Field", (int*)&isosurfaceType,
                ISOSURFACE_TYPE_NAMES, IM_ARRAYSIZE(ISOSURFACE_TYPE_NAMES))) {
            setShaderDirty();
            setGridData();
            reRender = true;
            frameInfo.frameCount = 0;
        }
        if (useIsosurfaces && propertyEditor.addCombo(
                "Surface BRDF", (int*)&surfaceBrdf, SURFACE_BRDF_NAMES, IM_ARRAYSIZE(SURFACE_BRDF_NAMES))) {
            setShaderDirty();
            reRender = true;
            frameInfo.frameCount = 0;
        }

        propertyEditor.endNode();
    }

    int numDenoisersSupported = IM_ARRAYSIZE(DENOISER_NAMES);
#ifdef SUPPORT_OPTIX
    if (!OptixVptDenoiser::isOptixEnabled()) {
        numDenoisersSupported--;
    }
#endif
    if (propertyEditor.addCombo(
            "Denoiser", (int*)&denoiserType, DENOISER_NAMES, numDenoisersSupported)) {
        denoiserChanged = true;
        reRender = true;
        changedDenoiserSettings = true;
    }

    if (useDenoiser && denoiser) {
        if (propertyEditor.beginNode(denoiser->getDenoiserName())) {
            bool denoiserReRender = denoiser->renderGuiPropertyEditorNodes(propertyEditor);
            reRender = denoiserReRender || reRender;
            changedDenoiserSettings = denoiserReRender || changedDenoiserSettings;
            if (denoiserReRender) {
                checkRecreateFeatureMaps();
            }
            propertyEditor.endNode();
        }
    }

    if (cloudData) {
        auto xs = cloudData->getGridSizeX();
        auto ys = cloudData->getGridSizeY();
        auto zs = cloudData->getGridSizeZ();
        propertyEditor.addText(
                "Volume Size", std::to_string(xs) + "x" + std::to_string(ys) + "x" + std::to_string(zs));
    }

    if (optionChanged) {
        frameInfo.frameCount = 0;
        reRender = true;
    }

    return optionChanged;
}


void buildCameraPrimitives(
        std::vector<uint32_t>& lineIndices,
        std::vector<glm::vec3>& vertexPositions,
        const std::vector<CameraPose>& cameraPoses) {
    const float dist = 0.01f;
    for (const auto& camPose : cameraPoses) {
        auto indexOffset = uint32_t(vertexPositions.size());
        const glm::vec3& pos = camPose.position;
        const glm::vec3& front = camPose.front;
        const glm::vec3& right = camPose.right;
        const glm::vec3& up = camPose.up;
        const float& fovy = camPose.fovy;
        const float& viewportWidth = camPose.viewportWidth;
        const float& viewportHeight = camPose.viewportHeight;
        float distUp = 2.0f * dist * std::tan(fovy * 0.5f);
        float distRight = viewportWidth / viewportHeight * distUp;
        vertexPositions.push_back(pos);
        vertexPositions.push_back(pos + dist * front - distRight * right - distUp * up);
        vertexPositions.push_back(pos + dist * front + distRight * right - distUp * up);
        vertexPositions.push_back(pos + dist * front + distRight * right + distUp * up);
        vertexPositions.push_back(pos + dist * front - distRight * right + distUp * up);
        for (uint32_t i = 0; i < 4; i++) {
            lineIndices.push_back(indexOffset);
            lineIndices.push_back(indexOffset + i + 1);
        }
        for (uint32_t i = 0; i < 4; i++) {
            lineIndices.push_back(indexOffset + i + 1);
            lineIndices.push_back(indexOffset + (i + 1) % 4 + 1);
        }
    }
}

void VolumetricPathTracingPass::setCameraPoses(const std::vector<CameraPose>& cameraPoses) {
    std::vector<uint32_t> lineIndices;
    std::vector<glm::vec3> vertexPositions;
    buildCameraPrimitives(lineIndices, vertexPositions, cameraPoses);
    cameraPoseLinePass->setRenderData(lineIndices, vertexPositions);
    if (!cameraPosesSet) {
        cameraPosesSet = true;
    }
}



BlitMomentTexturePass::BlitMomentTexturePass(sgl::vk::Renderer* renderer, std::string prefix)
        : BlitRenderPass(renderer, {"BlitMomentTexture.Vertex", "BlitMomentTexture.Fragment"}),
          prefix(std::move(prefix)) {
}

// Public interface.
void BlitMomentTexturePass::setOutputImage(sgl::vk::ImageViewPtr& colorImage) {
    BlitRenderPass::setOutputImage(colorImage);
    recreateMomentTexture();
}

void BlitMomentTexturePass::setVisualizeMomentTexture(bool visualizeMomentTexture) {
    this->visualizeMomentTexture = visualizeMomentTexture;
}

void BlitMomentTexturePass::recreateMomentTexture() {
    sgl::vk::ImageSamplerSettings samplerSettings;
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();

    if (momentType != MomentType::NONE) {
        sgl::vk::ImageSettings imageSettings = outputImageViews.front()->getImage()->getImageSettings();
        imageSettings.format = VK_FORMAT_R32_SFLOAT;
        imageSettings.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageSettings.arrayLayers = numMoments + 1;
        momentTexture = std::make_shared<sgl::vk::Texture>(
                device, imageSettings, VK_IMAGE_VIEW_TYPE_2D_ARRAY, samplerSettings);
        BlitRenderPass::setInputTexture(momentTexture);
    } else {
        momentTexture = {};
        inputTexture = {};
        rasterData = {};
        setDataDirty();
    }
}

bool BlitMomentTexturePass::renderGuiPropertyEditorNodes(
        sgl::PropertyEditor& propertyEditor, bool& shallRecreateMomentTexture, bool& momentTypeChanged) {
    bool reRender = false;
    shallRecreateMomentTexture = false;
    momentTypeChanged = false;

    std::string headerName = prefix;
    if (propertyEditor.beginNode(headerName)) {
        std::string momentTypeName = "Moment Type## (" + prefix + ")";
        MomentType momentTypeOld = momentType;
        if (propertyEditor.addCombo(
                momentTypeName, (int*)&momentTypeOld, MOMENT_TYPE_NAMES,
                IM_ARRAYSIZE(MOMENT_TYPE_NAMES))) {
            reRender = true;
            momentTypeChanged = true;
            if ((momentType == MomentType::NONE) != (momentTypeOld == MomentType::NONE)) {
                shallRecreateMomentTexture = true;
            }
            momentType = momentTypeOld;
        }
        std::string numMomentName = "#Moments## (" + prefix + ")";
        if (propertyEditor.addCombo(
                numMomentName, &numMomentsIdx, NUM_MOMENTS_NAMES,
                IM_ARRAYSIZE(NUM_MOMENTS_NAMES))) {
            numMoments = NUM_MOMENTS_SUPPORTED[numMomentsIdx];
            reRender = true;
            shallRecreateMomentTexture = true;
        }

        if (visualizeMomentTexture) {
            if (propertyEditor.addSliderInt("Visualized Moment", &selectedMomentBlitIdx, 0, numMoments)) {
                reRender = true;
            }
        }

        propertyEditor.endNode();
    }

    if (shallRecreateMomentTexture) {
        selectedMomentBlitIdx = std::min(selectedMomentBlitIdx, numMoments);
        recreateMomentTexture();
    }

    return reRender;
}

void BlitMomentTexturePass::createRasterData(sgl::vk::Renderer* renderer, sgl::vk::GraphicsPipelinePtr& graphicsPipeline) {
    BlitRenderPass::createRasterData(renderer, graphicsPipeline);
}

void BlitMomentTexturePass::renderOptional() {
    if (!momentTexture) {
        renderer->transitionImageLayout(
                outputImageViews.front()->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        outputImageViews.front()->clearColor(
                glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), renderer->getVkCommandBuffer());
        return;
    }

    render();
}

void BlitMomentTexturePass::_render() {
    renderer->transitionImageLayout(momentTexture->getImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    renderer->pushConstants(
            rasterData->getGraphicsPipeline(), VK_SHADER_STAGE_FRAGMENT_BIT,
            0, selectedMomentBlitIdx);
    BlitRenderPass::_render();
    renderer->transitionImageLayout(momentTexture->getImage(), VK_IMAGE_LAYOUT_GENERAL);
}


OctahedralMappingPass::OctahedralMappingPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
}

void OctahedralMappingPass::setInputImage(const sgl::vk::TexturePtr& _inputImage) {
    inputImage = _inputImage;
    setDataDirty();
}

void OctahedralMappingPass::setOutputImage(sgl::vk::ImageViewPtr& _outputImage) {
    outputImage = _outputImage;
    setDataDirty();
}

void OctahedralMappingPass::loadShader() {
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE", std::to_string(BLOCK_SIZE)));
    shaderStages = sgl::vk::ShaderManager->getShaderStages(
            { "OctahedralMapper.Compute" }, preprocessorDefines);
}

void OctahedralMappingPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticTexture(inputImage, "environmentMapTexture");
    computeData->setStaticImageView(outputImage, "outputImage");
}

void OctahedralMappingPass::_render() {
    auto width = int(outputImage->getImage()->getImageSettings().width);
    auto height = int(outputImage->getImage()->getImageSettings().height);
    renderer->dispatch(
            computeData,
            sgl::iceil(width, BLOCK_SIZE), sgl::iceil(height, BLOCK_SIZE), 1);
}
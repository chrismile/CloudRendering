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
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Render/ComputePipeline.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/MultiVarTransferFunctionWindow.hpp>

#include "CloudData.hpp"
#include "OccupancyGrid.hpp"

OccupancyGridPass::OccupancyGridPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);

    subgroupSize = device->getPhysicalDeviceSubgroupProperties().subgroupSize;
}

void OccupancyGridPass::renderIfNecessary() {
    if (isDirty) {
        render();
        isDirty = false;
    }
}

const sgl::vk::ImageViewPtr& OccupancyGridPass::getOccupancyGridImage() {
    if (!occupancyGridImage) {
        sgl::vk::ImageSettings imageSettings{};
        imageSettings.width = sgl::uiceil(config.cloudData->getGridSizeX(), occupancyGridSize);
        imageSettings.height = sgl::uiceil(config.cloudData->getGridSizeY(), occupancyGridSize);
        imageSettings.depth = sgl::uiceil(config.cloudData->getGridSizeZ(), occupancyGridSize);
        imageSettings.imageType = VK_IMAGE_TYPE_3D;
        imageSettings.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
        imageSettings.format = VK_FORMAT_R8_UINT;
        occupancyGridImage = std::make_shared<sgl::vk::ImageView>(std::make_shared<sgl::vk::Image>(
                device, imageSettings));
    }
    return occupancyGridImage;
}

void OccupancyGridPass::setConfig(const OccupancyGridConfig& _config) {
    if (config.cloudData != _config.cloudData) {
        setDataDirty();
        isDirty = true;
        occupancyGridImage = {}; //< Will be recreated in getOccupancyGridImage.
    }
    if (config.useSparseGrid != _config.useSparseGrid
            || config.useIsosurfaces != _config.useIsosurfaces
            || config.isosurfaceType != _config.isosurfaceType) {
        setShaderDirty();
        isDirty = true;
    }
    if (config.densityFieldTexture != _config.densityFieldTexture
            || config.nanoVdbBuffer != _config.nanoVdbBuffer
            || config.densityGradientFieldTexture != _config.densityGradientFieldTexture) {
        setDataDirty();
        isDirty = true;
    }
    if (config.isoValue != _config.isoValue
            || config.voxelValueMin != _config.voxelValueMin
            || config.voxelValueMax != _config.voxelValueMax
            || config.minGradientVal != _config.minGradientVal
            || config.maxGradientVal != _config.maxGradientVal) {
        uniformData.voxelValueMin = config.voxelValueMin;
        uniformData.voxelValueMax = config.voxelValueMax;
        uniformData.isoValue = config.isoValue;
        isDirty = true;
    }
    config = _config;
}

void OccupancyGridPass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("SUBGROUP_SIZE", std::to_string(subgroupSize)));

    if (config.useSparseGrid) {
        preprocessorDefines.insert({ "USE_NANOVDB", "" });
    }

    auto* tfWindow = config.cloudData->getTransferFunctionWindow();
    bool useTransferFunction = tfWindow && tfWindow->getShowWindow();
    if (useTransferFunction) {
        preprocessorDefines.insert({ "USE_TRANSFER_FUNCTION", "" });
    }

    if (config.useIsosurfaces) {
        preprocessorDefines.insert({ "USE_ISOSURFACES", "" });
        if (config.isosurfaceType == IsosurfaceType::DENSITY) {
            preprocessorDefines.insert({ "ISOSURFACE_TYPE_DENSITY", "" });
        } else if (config.isosurfaceType == IsosurfaceType::GRADIENT) {
            preprocessorDefines.insert({ "ISOSURFACE_TYPE_GRADIENT", "" });
        }
    }

    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "ComputeOccupancyGrid.Compute" }, preprocessorDefines);
}

void OccupancyGridPass::createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    if (config.useSparseGrid) {
        computeData->setStaticBuffer(config.nanoVdbBuffer, "NanoVdbBuffer");
    } else {
        computeData->setStaticTexture(config.densityFieldTexture, "gridImage");
        if (config.useIsosurfaces && config.isosurfaceType == IsosurfaceType::GRADIENT) {
            computeData->setStaticTexture(config.densityGradientFieldTexture, "gradientImage");
        }
    }
    auto* tfWindow = config.cloudData->getTransferFunctionWindow();
    if (tfWindow && tfWindow->getShowWindow()) {
        computeData->setStaticTexture(tfWindow->getTransferFunctionMapTextureVulkan(), "transferFunctionTexture");
    }
    computeData->setStaticBuffer(uniformBuffer, "Parameters");
    computeData->setStaticImageView(occupancyGridImage, "occupancyGridImage");
}

void OccupancyGridPass::_render() {
    /*
     * At the moment, no further layout transitions are necessary, as empty super voxels are also updated by the compute
     * shader and no initial clear pass with TRANSFER_DST layout is thus needed.
     */
    if (occupancyGridImage->getImage()->getVkImageLayout() == VK_IMAGE_LAYOUT_UNDEFINED) {
        renderer->insertImageMemoryBarrier(
                occupancyGridImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);
    }

    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertBufferMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            uniformBuffer);

    const auto& occGridSettings = occupancyGridImage->getImage()->getImageSettings();
    auto numBlocks = occGridSettings.width * occGridSettings.height * occGridSettings.depth;
    renderer->pushConstants(
            computeData->getComputePipeline(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
            glm::uvec2(occupancyGridSize, numBlocks));
    renderer->dispatch(computeData, numBlocks, 1, 1);
    renderer->insertImageMemoryBarrier(
            occupancyGridImage,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
}

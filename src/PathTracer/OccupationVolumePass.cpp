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

#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/Widgets/MultiVarTransferFunctionWindow.hpp>

#include "CloudData.hpp"
#include "VolumetricPathTracingPass.hpp"
#include "OccupationVolumePass.hpp"

class MaxFilterPass : public sgl::vk::ComputePass {
public:
    explicit MaxFilterPass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {}
    ~MaxFilterPass() override {}
    void setData(
            const sgl::vk::ImageViewPtr& imageIn, const sgl::vk::ImageViewPtr& imageOut, uint32_t _maxKernelRadius) {
        if (maxKernelRadius != _maxKernelRadius) {
            maxKernelRadius = _maxKernelRadius;
            setShaderDirty();
        }
        occupationVolumeImageIn = imageIn;
        occupationVolumeImageOut = imageOut;
        setDataDirty();
    }

protected:
    void loadShader() override {
        sgl::vk::ShaderManager->invalidateShaderCache();
        std::map<std::string, std::string> preprocessorDefines;
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(BLOCK_SIZE_X)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(BLOCK_SIZE_Y)));
        preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(BLOCK_SIZE_Z)));

        preprocessorDefines.insert({ "FILTER_RADIUS", std::to_string(maxKernelRadius) });

        shaderStages = sgl::vk::ShaderManager->getShaderStages(
                { "OccupationVolume.MaxKernel.Compute" }, preprocessorDefines);
    }
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override {
        computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
        computeData->setStaticImageView(occupationVolumeImageIn, "occupationVolumeImageIn");
        computeData->setStaticImageView(occupationVolumeImageOut, "occupationVolumeImageOut");
    }
    void _render() override {
        auto& imageSettings = occupationVolumeImageIn->getImage()->getImageSettings();
        renderer->dispatch(
                computeData,
                sgl::uiceil(imageSettings.width, BLOCK_SIZE_X),
                sgl::uiceil(imageSettings.height, BLOCK_SIZE_Y),
                sgl::uiceil(imageSettings.depth, BLOCK_SIZE_Z));
    }

    const uint32_t BLOCK_SIZE_X = 8;
    const uint32_t BLOCK_SIZE_Y = 8;
    const uint32_t BLOCK_SIZE_Z = 4;

private:
    sgl::vk::ImageViewPtr occupationVolumeImageIn;
    sgl::vk::ImageViewPtr occupationVolumeImageOut;
    uint32_t maxKernelRadius = 1;
};



OccupationVolumePass::OccupationVolumePass(sgl::vk::Renderer* renderer) : ComputePass(renderer) {
    uniformBuffer = std::make_shared<sgl::vk::Buffer>(
            device, sizeof(UniformData),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
}

OccupationVolumePass::~OccupationVolumePass() {
    if (maxFilterPass) {
        delete maxFilterPass;
        maxFilterPass = nullptr;
    }
}

sgl::vk::ImageViewPtr OccupationVolumePass::computeVolume(
        VolumetricPathTracingPass* vptPass, uint32_t _ds, uint32_t maxKernelRadius) {
    auto cloudData = vptPass->cloudData;
    ds = _ds;

    useSparseGrid = vptPass->useSparseGrid;
    if (vptPass->useSparseGrid) {
        nanoVdbBuffer = vptPass->nanoVdbBuffer;
    } else {
        densityFieldTexture = vptPass->densityFieldTexture;
    }

    tfWindow = cloudData->getTransferFunctionWindow();

    useIsosurfaces = vptPass->useIsosurfaces;
    isoValue = vptPass->isoValue;
    isosurfaceType = vptPass->isosurfaceType;
    if (isosurfaceType == IsosurfaceType::GRADIENT) {
        densityGradientFieldTexture = vptPass->densityGradientFieldTexture;
    }

    occupationVolumeImage = {};
    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = sgl::uiceil(cloudData->getGridSizeX(), ds);
    imageSettings.height = sgl::uiceil(cloudData->getGridSizeY(), ds);
    imageSettings.depth = sgl::uiceil(cloudData->getGridSizeZ(), ds);
    imageSettings.imageType = VK_IMAGE_TYPE_3D;
    imageSettings.format = VK_FORMAT_R8_UINT;
    imageSettings.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    occupationVolumeImage = std::make_shared<sgl::vk::ImageView>(std::make_shared<sgl::vk::Image>(device, imageSettings));

    uniformData.isoValue = isoValue;
    if (!useSparseGrid) {
        auto densityField = cloudData->getDenseDensityField();
        uniformData.voxelValueMin = densityField->getMinValue();
        uniformData.voxelValueMax = densityField->getMaxValue();
    } else {
        uniformData.voxelValueMin = 0.0f;
        uniformData.voxelValueMax = 1.0f;
    }
    uniformBuffer->updateData(
            sizeof(UniformData), &uniformData, renderer->getVkCommandBuffer());
    renderer->insertMemoryBarrier(
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    renderer->insertImageMemoryBarrier(
            occupationVolumeImage,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);
    render();

    if (maxKernelRadius > 1) {
        if (!maxFilterPass) {
            maxFilterPass = new MaxFilterPass(renderer);
        }
        auto occupationVolumeImageIn = occupationVolumeImage;
        sgl::vk::ImageSettings imageSettings{};
        imageSettings.width = sgl::uiceil(cloudData->getGridSizeX(), ds);
        imageSettings.height = sgl::uiceil(cloudData->getGridSizeY(), ds);
        imageSettings.depth = sgl::uiceil(cloudData->getGridSizeZ(), ds);
        imageSettings.imageType = VK_IMAGE_TYPE_3D;
        imageSettings.format = VK_FORMAT_R8_UINT;
        imageSettings.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        occupationVolumeImage = std::make_shared<sgl::vk::ImageView>(std::make_shared<sgl::vk::Image>(device, imageSettings));

        renderer->insertImageMemoryBarrier(
                occupationVolumeImage,
                VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_NONE_KHR, VK_ACCESS_SHADER_WRITE_BIT);
        renderer->insertImageMemoryBarrier(
                occupationVolumeImageIn,
                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        maxFilterPass->setData(occupationVolumeImageIn, occupationVolumeImage, maxKernelRadius);
        maxFilterPass->render();
    }

    renderer->insertImageMemoryBarrier(
            occupationVolumeImage,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    return occupationVolumeImage;
}

void OccupationVolumePass::loadShader() {
    sgl::vk::ShaderManager->invalidateShaderCache();
    std::map<std::string, std::string> preprocessorDefines;
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_X", std::to_string(BLOCK_SIZE_X)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Y", std::to_string(BLOCK_SIZE_Y)));
    preprocessorDefines.insert(std::make_pair("BLOCK_SIZE_Z", std::to_string(BLOCK_SIZE_Z)));

    preprocessorDefines.insert({ "SUBSAMPLING_FACTOR", std::to_string(ds) });

    if (useSparseGrid) {
        preprocessorDefines.insert({ "USE_NANOVDB", "" });
    }

    if (tfWindow && tfWindow->getShowWindow()) {
        preprocessorDefines.insert({ "USE_TRANSFER_FUNCTION", "" });
    }

    if (useIsosurfaces) {
        preprocessorDefines.insert({ "USE_ISOSURFACES", "" });
        if (isosurfaceType == IsosurfaceType::DENSITY) {
            preprocessorDefines.insert({ "ISOSURFACE_TYPE_DENSITY", "" });
        } else if (isosurfaceType == IsosurfaceType::GRADIENT) {
            preprocessorDefines.insert({ "ISOSURFACE_TYPE_GRADIENT", "" });
        }
    }

    shaderStages = sgl::vk::ShaderManager->getShaderStages({ "OccupationVolume.Compute" }, preprocessorDefines);
}

void OccupationVolumePass::createComputeData(
        sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) {
    computeData = std::make_shared<sgl::vk::ComputeData>(renderer, computePipeline);
    computeData->setStaticImageView(occupationVolumeImage, "occupationVolumeImage");
    computeData->setStaticBuffer(uniformBuffer, "UniformBuffer");

    if (useSparseGrid) {
        computeData->setStaticBuffer(nanoVdbBuffer, "NanoVdbBuffer");
    } else {
        computeData->setStaticTexture(densityFieldTexture, "gridImage");
        if (useIsosurfaces && isosurfaceType == IsosurfaceType::GRADIENT) {
            computeData->setStaticTexture(densityGradientFieldTexture, "gradientImage");
        }
    }

    if (tfWindow && tfWindow->getShowWindow()) {
        computeData->setStaticTexture(tfWindow->getTransferFunctionMapTextureVulkan(), "transferFunctionTexture");
    }
}

void OccupationVolumePass::_render() {
    auto& imageSettings = occupationVolumeImage->getImage()->getImageSettings();
    renderer->dispatch(
            computeData,
            sgl::uiceil(imageSettings.width, BLOCK_SIZE_X),
            sgl::uiceil(imageSettings.height, BLOCK_SIZE_Y),
            sgl::uiceil(imageSettings.depth, BLOCK_SIZE_Z));
}

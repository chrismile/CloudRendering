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

#ifndef CLOUDRENDERING_OCCUPATIONVOLUMEPASS_HPP
#define CLOUDRENDERING_OCCUPATIONVOLUMEPASS_HPP

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "RenderSettings.hpp"

namespace sgl {
class MultiVarTransferFunctionWindow;
}

class CloudData;
typedef std::shared_ptr<CloudData> CloudDataPtr;
class VolumetricPathTracingPass;
class MaxFilterPass;

class OccupationVolumePass : public sgl::vk::ComputePass {
public:
    explicit OccupationVolumePass(sgl::vk::Renderer* renderer);
    ~OccupationVolumePass() override;

    sgl::vk::ImageViewPtr computeVolume(VolumetricPathTracingPass* vptPass, uint32_t _ds, uint32_t maxKernelRadius);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

    const uint32_t BLOCK_SIZE_X = 8;
    const uint32_t BLOCK_SIZE_Y = 8;
    const uint32_t BLOCK_SIZE_Z = 4;

private:
    uint32_t ds = 1; ///< Downsampling factor.
    sgl::MultiVarTransferFunctionWindow* tfWindow;
    MaxFilterPass* maxFilterPass = nullptr;

    bool useSparseGrid = false; ///< Use NanoVDB or a dense grid texture?
    sgl::vk::TexturePtr densityFieldTexture; /// < Dense grid texture.
    sgl::vk::BufferPtr nanoVdbBuffer; /// < Sparse grid buffer.

    /// Optional; only for isosurfaceType == IsosurfaceType::GRADIENT.
    sgl::vk::TexturePtr densityGradientFieldTexture;

    sgl::vk::ImageViewPtr occupationVolumeImage;

    bool useIsosurfaces = false;
    float isoValue = 0.5f;
    IsosurfaceType isosurfaceType = IsosurfaceType::DENSITY;
    struct UniformData {
        float voxelValueMin = 0.0f;
        float voxelValueMax = 1.0f;
        float isoValue = 0.5f;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;
};

#endif //CLOUDRENDERING_OCCUPATIONVOLUMEPASS_HPP

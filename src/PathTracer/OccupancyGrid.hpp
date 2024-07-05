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

#ifndef CLOUDRENDERING_OCCUPANCYGRID_HPP
#define CLOUDRENDERING_OCCUPANCYGRID_HPP

#include <glm/vec3.hpp>

#include <Graphics/Vulkan/Render/Passes/Pass.hpp>

#include "RenderSettings.hpp"

class CloudData;
typedef std::shared_ptr<CloudData> CloudDataPtr;

struct OccupancyGridConfig {
    CloudDataPtr cloudData;
    bool useSparseGrid = false; ///< Use NanoVDB or a dense grid texture?

    sgl::vk::TexturePtr densityFieldTexture; /// < Dense grid texture.
    sgl::vk::BufferPtr nanoVdbBuffer; /// < Sparse grid buffer.
    /// Optional; only for isosurfaceType == IsosurfaceType::GRADIENT.
    sgl::vk::TexturePtr densityGradientFieldTexture;

    bool useIsosurfaces = false;
    float isoValue = 0.5f;
    IsosurfaceType isosurfaceType = IsosurfaceType::DENSITY;
    float voxelValueMin = 0.0f, voxelValueMax = 1.0f;
    float minGradientVal = 0.0f, maxGradientVal = 1.0f;
};

class OccupancyGridPass : public sgl::vk::ComputePass {
public:
    explicit OccupancyGridPass(sgl::vk::Renderer* renderer);
    void renderIfNecessary();
    const sgl::vk::ImageViewPtr& getOccupancyGridImage();
    [[nodiscard]] inline uint32_t getBlockSize() const { return occupancyGridSize; }
    [[nodiscard]] inline glm::ivec3 getSuperVoxelSize() const { return glm::ivec3(int(getBlockSize())); }
    [[nodiscard]] inline glm::ivec3 getSuperVoxelGridSize() const {
        const auto& imageSettings = occupancyGridImage->getImage()->getImageSettings();
        return { int(imageSettings.width), int(imageSettings.height), int(imageSettings.depth) };
    }

    void setConfig(const OccupancyGridConfig& config);

protected:
    void loadShader() override;
    void createComputeData(sgl::vk::Renderer* renderer, sgl::vk::ComputePipelinePtr& computePipeline) override;
    void _render() override;

private:
    uint32_t subgroupSize;
    const uint32_t occupancyGridSize = 8; // In 3D: 8*8*8.

    sgl::vk::ImageViewPtr occupancyGridImage;
    bool isDirty = true;
    OccupancyGridConfig config{};

    struct UniformData {
        // Cloud properties.
        float voxelValueMin;
        float voxelValueMax;

        // Isosurfaces.
        float isoValue = 0.5f;
        float pad0;
    };
    UniformData uniformData{};
    sgl::vk::BufferPtr uniformBuffer;
};

#endif //CLOUDRENDERING_OCCUPANCYGRID_HPP

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

#include <iostream>

#include <Utils/Mesh/IndexMesh.hpp>

#include <IsosurfaceCpp/src/MarchingCubes.hpp>
#include <IsosurfaceCpp/src/SnapMC.hpp>

#include "Utils/Normalization.hpp"
#include "CloudData.hpp"

void CloudData::createIsoSurfaceData(
        const IsosurfaceSettings& settings,
        std::vector<uint32_t>& triangleIndices, std::vector<glm::vec3>& vertexPositions,
        std::vector<glm::vec4>& vertexColors, std::vector<glm::vec3>& vertexNormals) {
    auto densityFieldExport = getDenseDensityField();
    auto dx = float(voxelSizeX);
    auto dy = float(voxelSizeY);
    auto dz = float(voxelSizeZ);

    sgl::AABB3 gridAabb;
    //gridAabb.min = glm::vec3(0.0f, 0.0f, 0.0f);
    //gridAabb.max = glm::vec3(gridSizeX, gridSizeY, gridSizeZ);
    //gridAabb.min = glm::vec3(-0.5f, -0.5f, -0.5f);
    //gridAabb.max = glm::vec3(gridSizeX, gridSizeY, gridSizeZ) - glm::vec3(0.5f, 0.5f, 0.5f);
    gridAabb.min = glm::vec3(-0.5f, -0.5f, -0.5f);
    gridAabb.max = glm::vec3(gridSizeX, gridSizeY, gridSizeZ) - glm::vec3(0.5f, 0.5f, 0.5f);
    gridAabb.min *= glm::vec3(dx, dy, dz);
    gridAabb.max *= glm::vec3(dx, dy, dz);

    float minVal = densityFieldExport->getMinValue();
    float maxVal = densityFieldExport->getMaxValue();
    float isoValueCorrected = settings.isoValue;
    if ((minVal != 0.0f || maxVal != 1.0f) && minVal != maxVal) {
        isoValueCorrected = isoValueCorrected * (maxVal - minVal) - minVal;
    }

    std::vector<glm::vec3> isosurfaceVertexPositions;
    std::vector<glm::vec3> isosurfaceVertexNormals;
    if (settings.isoSurfaceExtractionTechnique == IsoSurfaceExtractionTechnique::MARCHING_CUBES) {
        polygonizeMarchingCubes(
                densityFieldExport->data<float>(),
                int(gridSizeX), int(gridSizeY), int(gridSizeZ), dx, dy, dz,
                isoValueCorrected, isosurfaceVertexPositions, isosurfaceVertexNormals);
    } else {
        polygonizeSnapMC(
                densityFieldExport->data<float>(),
                int(gridSizeX), int(gridSizeY), int(gridSizeZ), dx, dy, dz,
                isoValueCorrected, settings.gammaSnapMC, isosurfaceVertexPositions, isosurfaceVertexNormals);
    }

    float step = std::min(dx, std::min(dy, dz));
    sgl::computeSharedIndexRepresentation(
            isosurfaceVertexPositions, isosurfaceVertexNormals,
            triangleIndices, vertexPositions, vertexNormals,
            1e-5f * step);
    normalizeVertexPositions(vertexPositions, gridAabb, nullptr);
    //normalizeVertexNormals(vertexNormals, gridAabb, nullptr);

    // TODO: Compute vertex colors from transfer function if settings.useIsosurfaceTf.
    for (size_t i = 0; i < vertexPositions.size(); i++) {
        vertexColors.emplace_back(settings.isosurfaceColor);
    }
}

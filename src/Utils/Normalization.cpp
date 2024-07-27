/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include "Normalization.hpp"

void normalizeVertexPosition(
        glm::vec3& vertexPosition, const sgl::AABB3& aabb,
        const glm::mat4* vertexTransformationMatrixPtr) {
    glm::vec3 translation = -aabb.getCenter();
    glm::vec3 scale3D = 0.5f / aabb.getDimensions();
    float scale = std::min(scale3D.x, std::min(scale3D.y, scale3D.z));

    glm::vec3& v = vertexPosition;
    v = (v + translation) * scale;

    if (vertexTransformationMatrixPtr != nullptr) {
        glm::mat4 transformationMatrix = *vertexTransformationMatrixPtr;
        glm::vec4 transformedVec = transformationMatrix * glm::vec4(v.x, v.y, v.z, 1.0f);
        v = glm::vec3(transformedVec.x, transformedVec.y, transformedVec.z);
    }
}

void normalizeVertexPositions(
        std::vector<glm::vec3>& vertexPositions, const sgl::AABB3& aabb,
        const glm::mat4* vertexTransformationMatrixPtr) {
    glm::vec3 translation = -aabb.getCenter();
    glm::vec3 scale3D = 0.5f / aabb.getDimensions();
    float scale = std::min(scale3D.x, std::min(scale3D.y, scale3D.z));

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, vertexPositions.size()), [&](auto const& r) {
        for (size_t vertexIdx = r.begin(); vertexIdx != r.end(); vertexIdx++) {
#else
#if _OPENMP >= 200805
    #pragma omp parallel for shared(vertexPositions, translation, scale) default(none)
#endif
    for (size_t vertexIdx = 0; vertexIdx < vertexPositions.size(); vertexIdx++) {
#endif
        glm::vec3& v = vertexPositions.at(vertexIdx);
        v = (v + translation) * scale;
    }
#ifdef USE_TBB
    });
#endif

    if (vertexTransformationMatrixPtr != nullptr) {
        glm::mat4 transformationMatrix = *vertexTransformationMatrixPtr;

#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, vertexPositions.size()), [&](auto const& r) {
            for (size_t vertexIdx = r.begin(); vertexIdx != r.end(); vertexIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel for shared(vertexPositions, transformationMatrix) default(none)
#endif
        for (size_t vertexIdx = 0; vertexIdx < vertexPositions.size(); vertexIdx++) {
#endif
            glm::vec3& v = vertexPositions.at(vertexIdx);
            glm::vec4 transformedVec = transformationMatrix * glm::vec4(v.x, v.y, v.z, 1.0f);
            v = glm::vec3(transformedVec.x, transformedVec.y, transformedVec.z);
        }
#ifdef USE_TBB
        });
#endif
    }
}

void normalizeVertexNormals(
        std::vector<glm::vec3>& vertexNormals, const sgl::AABB3& aabb,
        const glm::mat4* vertexTransformationMatrixPtr) {
    if (vertexTransformationMatrixPtr != nullptr) {
        glm::mat4 transformationMatrix = *vertexTransformationMatrixPtr;
        transformationMatrix = glm::transpose(glm::inverse(transformationMatrix));

#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, vertexNormals.size()), [&](auto const& r) {
            for (size_t vertexIdx = r.begin(); vertexIdx != r.end(); vertexIdx++) {
#else
#if _OPENMP >= 200805
        #pragma omp parallel for shared(vertexNormals, transformationMatrix) default(none)
#endif
        for (size_t vertexIdx = 0; vertexIdx < vertexNormals.size(); vertexIdx++) {
#endif
            glm::vec3& v = vertexNormals.at(vertexIdx);
            glm::vec4 transformedVec = transformationMatrix * glm::vec4(v.x, v.y, v.z, 0.0f);
            v = glm::vec3(transformedVec.x, transformedVec.y, transformedVec.z);
        }
#ifdef USE_TBB
        });
#endif
    }
}

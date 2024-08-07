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
#include <Math/Math.hpp>
#include "EnergyTerm.cuh"

#define F_PI 3.1415926535897932f
#define F_TWO_PI (2.0f * F_PI)

__device__ float sign(float x) {
    return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f);
}

__device__ int iclamp(int x, int l, int u) {
    return x <= l ? l : (x >= u ? u : x);
}

__global__ void updateObservationFrequencyFieldsKernel(
        uint32_t depth, uint32_t height, uint32_t width, uint32_t numBinsX, uint32_t numBinsY, uint32_t numBins,
        float camPosX, float camPosY, float camPosZ, const float* __restrict__ transmittanceField,
        float* __restrict__ obsFreqField, float* __restrict__ angularObsFreqField) {
    uint32_t idX = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idY = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idZ = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t linearIdx = idX + (idY + idZ * height) * width;
    if (idX >= width || idY >= height || idZ >= depth) {
        return;
    }
    float transmittance = transmittanceField[linearIdx];

    float dx = float(idX) - camPosX;
    float dy = float(idY) - camPosY;
    float dz = float(idZ) - camPosZ;
    float len = sqrtf(dx * dx + dy * dy + dz * dz);
    dx /= len;
    dy /= len;
    dz /= len;
    //float theta = acosf(dz);
    //float phi = sign(dy) * acosf(dx / sqrt(dx * dx + dy * dy));
    // https://mathworld.wolfram.com/SphericalCoordinates.html
    float theta = atan2f(dy, dx);
    float phi = acosf(dz);
    auto binIdxX = uint32_t(iclamp(int(fmodf(theta + F_TWO_PI, F_TWO_PI) / F_TWO_PI * float(numBinsX)), 0, int(numBinsX) - 1));
    auto binIdxY = uint32_t(iclamp(int(fmodf(phi + F_TWO_PI, F_PI) / F_PI * float(numBinsY)), 0, int(numBinsY) - 1));
    obsFreqField[linearIdx] += transmittance;
    angularObsFreqField[linearIdx * numBins + binIdxX + binIdxY * numBinsX] += transmittance;
}

#define CORRECT_AREA_PRESERVATION

__global__ void computeEnergyKernel(
        uint32_t depth, uint32_t height, uint32_t width, uint32_t numBinsX, uint32_t numBinsY, uint32_t numBins,
        uint32_t numCams, float gamma,
        const float* __restrict__ obsFreqField, const float* __restrict__ angularObsFreqField,
        const uint8_t* __restrict__ occupancyField, float* __restrict__ energyTermField) {
    uint32_t idX = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idY = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idZ = blockIdx.z * blockDim.z + threadIdx.z;
    uint32_t linearIdx = idX + (idY + idZ * height) * width;
    if (idX >= width || idY >= height || idZ >= depth) {
        return;
    }

    bool isOccupied = occupancyField[linearIdx] > 0u;
    if (isOccupied) {
        float energyTermLocal = 0.0f;
        const float fN = 1.0f / float(numCams);
        energyTermLocal += pow(obsFreqField[linearIdx] * fN, gamma);

        // Use total variation distance.
        float entrySum = 0.0f, TV = 0.0f;
#ifdef CORRECT_AREA_PRESERVATION
        float areaFactorSum = 0.0f;
#endif
        for (int binIdx = 0; binIdx < numBins; binIdx++) {
            entrySum += angularObsFreqField[linearIdx * numBins + binIdx];
#ifdef CORRECT_AREA_PRESERVATION
            uint32_t binIdxY = binIdx / numBinsX; // binIdxX + binIdxY * numBinsX
            float phiMid = (float(binIdxY) + 0.5f) / float(numBinsY) * F_PI;
            areaFactorSum += sinf(phiMid);
#endif
        }
        if (entrySum > 1e-6f) {
            float invEntrySum = 1.0f / entrySum;
            for (uint32_t binIdx = 0; binIdx < numBins; binIdx++) {
#ifdef CORRECT_AREA_PRESERVATION
                uint32_t binIdxY = binIdx / numBinsX; // binIdxX + binIdxY * numBinsX
                float phiMid = (float(binIdxY) + 0.5f) / float(numBinsY) * F_PI;
                TV +=
                        abs(angularObsFreqField[linearIdx * numBins + binIdx] * invEntrySum - invEntrySum)
                        * sinf(phiMid) / areaFactorSum;
#else
                TV += abs(angularObsFreqField[linearIdx * numBins + binIdx] * invEntrySum - invEntrySum);
#endif
            }
            energyTermLocal += 1.0f - TV * 0.5f;
        }
        energyTermField[linearIdx] = energyTermLocal;
    }
}

void updateObservationFrequencyFieldsImpl(
        cudaStream_t stream, uint32_t depth, uint32_t height, uint32_t width, uint32_t numBinsX, uint32_t numBinsY,
        const glm::vec3& camPos, const float* transmittanceField, float* obsFreqField, float* angularObsFreqField) {
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(sgl::uiceil(width, blockDim.x), sgl::uiceil(height, blockDim.y), sgl::uiceil(depth, blockDim.z));
    updateObservationFrequencyFieldsKernel<<<gridDim, blockDim, 0, stream>>> (
            depth, height, width, numBinsX, numBinsY, numBinsX * numBinsY, camPos.x, camPos.y, camPos.z,
            transmittanceField, obsFreqField, angularObsFreqField);
}

void computeEnergyImpl(
        cudaStream_t stream, uint32_t depth, uint32_t height, uint32_t width, uint32_t numBinsX, uint32_t numBinsY,
        uint32_t numCams, float gamma,
        const float* obsFreqField, const float* angularObsFreqField,
        const uint8_t* occupancyField, float* energyTermField) {
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(sgl::uiceil(width, blockDim.x), sgl::uiceil(height, blockDim.y), sgl::uiceil(depth, blockDim.z));
    computeEnergyKernel<<<gridDim, blockDim, 0, stream>>> (
            depth, height, width, numBinsX, numBinsY, numBinsX * numBinsY, numCams, gamma,
            obsFreqField, angularObsFreqField, occupancyField, energyTermField);
}

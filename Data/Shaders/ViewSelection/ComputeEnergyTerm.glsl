/**
 * MIT License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

-- Compute

#version 450

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout(push_constant) uniform PushConstants {
    uint numSamples;
    uint numBins;
};

layout (binding = 0, r8ui) readonly uniform uimage3D occupationVolumeImage;

layout (binding = 1) readonly buffer ObsFreqVolumeBuffer {
    float obsFreqField[];
};

layout (binding = 2) readonly buffer AngularObsFreqVolumeBuffer {
    float angularObsFreqField[];
};

layout (binding = 3) coherent buffer AngularObsFreqVolumeBuffer {
    float energyTerm;
};

#define M_PI 3.1415926535897932
#define TWO_PI (2.0 * M_PI)

void main() {
    uvec3 workSize = uvec3(imageSize(occupationVolumeImage));
    ivec3 imageCoord = ivec3(gl_GlobalInvocationID.xyz);
    uvec3 id = uvec3(gl_GlobalInvocationID.xyz);
    if (id.x >= workSize.x || id.y >= workSize.y || id.z >= workSize.z) {
        return;
    }

    bool isOccupied = imageLoad(occupationVolumeImage, imageCoord).r > 0u;
    if (isOccupied) {
        float energyTermLocal = 0.0;
        uint linearIdx = id.x + (id.y + id.z * workSize.y) * workSize.x;
        const float fN = 1.0 / float(numSamples);
        energyTermLocal += pow(obsFreqField[linearIdx] * fN, gamma);

        // Use total variation distance.
        float entrySum = 0.0f, TV = 0.0f;
        for (int binIdx = 0; binIdx < numBins; binIdx++) {
            entrySum += angularObsFreqField[linearIdx * numBins + binIdx];
        }
        if (entrySum > 1e-6) {
            float invEntrySum = 1.0 / entrySum;
            for (int binIdx = 0; binIdx < numBins; binIdx++) {
                TV += abs(angularObsFreqField[linearIdx * numBins + binIdx] * invEntrySum - invEntrySum);
            }
            energyTermLocal += 1.0 - TV * 0.5f;
        }

        atomicAdd(energyTerm, energyTermLocal);
    }
}

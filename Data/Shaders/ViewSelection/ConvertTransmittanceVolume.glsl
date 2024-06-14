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

layout (binding = 1, r32ui) uniform readonly uimage3D transmittanceVolumeImage;

layout (binding = 2) writeonly buffer TransmittanceVolumeBuffer {
    float transmittanceVolumeBuffer[];
};

#include "FloatPack.glsl"

void main() {
    uvec3 workSize = uvec3(imageSize(transmittanceVolumeImage));
    uvec3 id = uvec3(gl_GlobalInvocationID.xyz);
    if (id.x >= workSize.x || id.y >= workSize.y || id.z >= workSize.z) {
        return;
    }
    uint transmittanceUint = imageLoad(transmittanceVolumeImage, ivec3(id)).r;
    float transmittance = convertUint32ToNormalizedFloat(transmittanceUint);
    transmittanceVolumeBuffer[id.x + (id.y + id.z * workSize.y) * workSize.x] = transmittance;
}

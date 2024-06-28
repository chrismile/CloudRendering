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

//#extension GL_EXT_debug_printf : enable

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout (binding = 0, r8ui) writeonly uniform uimage3D occupationVolumeImage;

#ifdef USE_NANOVDB
// TODO: NanoVDB is not yet supported in the code below.
layout (binding = 1) readonly buffer NanoVdbBuffer {
    uint pnanovdb_buf_data[];
};
#else // USE_NANOVDB
layout (binding = 1) uniform sampler3D gridImage;
#endif // USE_NANOVDB

#if defined(ISOSURFACE_TYPE_GRADIENT)
layout(binding = 2) uniform sampler3D gradientImage;
#endif

#ifdef USE_TRANSFER_FUNCTION
layout(binding = 3) uniform sampler1DArray transferFunctionTexture;
#endif

layout (binding = 4) uniform UniformBuffer {
    float voxelValueMin;
    float voxelValueMax;

    // Isosurfaces.
    float isoValue;
} parameters;

#define sampleGridImage(coord) (texelFetch(gridImage, coord, 0).x - parameters.voxelValueMin) / (parameters.voxelValueMax - parameters.voxelValueMin)
float sampleCloud(in ivec3 voxelCoord) {
    return sampleGridImage(voxelCoord).r;
}

#ifdef USE_TRANSFER_FUNCTION
vec4 sampleCloudDensityEmission(in ivec3 voxelCoord) {
    // Idea: Returns (color.rgb, density).
    float densityRaw = sampleCloud(voxelCoord);
    //return texture(transferFunctionTexture, densityRaw);
    return texture(transferFunctionTexture, vec2(densityRaw, 0.0));
}
#endif

#if defined(ISOSURFACE_TYPE_DENSITY)
#define sampleCloudIso sampleCloud
#elif defined(ISOSURFACE_TYPE_GRADIENT)
float sampleCloudIso(in ivec3 voxelCoord) {
    return texelFetch(gradientImage, voxelCoord, 0).x;
}
#endif

uint getOccupation(ivec3 voxelCoord) {
#ifdef USE_TRANSFER_FUNCTION
    vec4 densityEmission = sampleCloudDensityEmission(voxelCoord);
    float density = densityEmission.a;
#else
    float density = sampleCloud(voxelCoord);
#endif

    uint occupation = 0u;
    if (density > 1e-4) {
        occupation = 1u;
    }

#ifdef USE_ISOSURFACES
    float volumeIsoValue = sampleCloudIso(voxelCoord);
    if (volumeIsoValue > parameters.isoValue) {
        occupation = 1u;
    }
#endif

    return occupation;
}

void main() {
    ivec3 dimIn = textureSize(gridImage, 0);
    ivec3 dimOut = imageSize(occupationVolumeImage);
    ivec3 imageCoord = ivec3(gl_GlobalInvocationID.xyz);
    if (imageCoord.x >= dimOut.x || imageCoord.y >= dimOut.y || imageCoord.z >= dimOut.z) {
        return;
    }

#if SUBSAMPLING_FACTOR == 1
    uint occupation = getOccupation(imageCoord);
#else
    uint occupation = 0u;
    ivec3 voxelOffset = imageCoord * ivec3(SUBSAMPLING_FACTOR, SUBSAMPLING_FACTOR, SUBSAMPLING_FACTOR);
    for (int z = 0; z < SUBSAMPLING_FACTOR; z++) {
        for (int y = 0; y < SUBSAMPLING_FACTOR; y++) {
            for (int x = 0; x < SUBSAMPLING_FACTOR; x++) {
                ivec3 voxelCoord = voxelOffset + ivec3(x, y, z);
                if (voxelCoord.x < dimIn.x && voxelCoord.y < dimIn.y && voxelCoord.z < dimIn.z) {
                    occupation = max(occupation, getOccupation(voxelCoord));
                }
            }
        }
    }
#endif

    imageStore(occupationVolumeImage, imageCoord, uvec4(occupation));
}


-- MaxKernel.Compute

#version 450

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = BLOCK_SIZE_Z) in;

layout (binding = 0, r8ui) readonly uniform uimage3D occupationVolumeImageIn;
layout (binding = 1, r8ui) writeonly uniform uimage3D occupationVolumeImageOut;

void main() {
    ivec3 dim = imageSize(occupationVolumeImageIn);
    ivec3 imageCoord = ivec3(gl_GlobalInvocationID.xyz);
    if (imageCoord.x >= dim.x || imageCoord.y >= dim.y || imageCoord.z >= dim.z) {
        return;
    }

    uint occupation = 0u;
    for (int z = -FILTER_RADIUS; z <= FILTER_RADIUS; z++) {
        for (int y = -FILTER_RADIUS; y <= FILTER_RADIUS; y++) {
            for (int x = -FILTER_RADIUS; x <= FILTER_RADIUS; x++) {
                ivec3 voxelCoord = imageCoord + ivec3(x, y, z);
                if (voxelCoord.x >= 0 && voxelCoord.y >= 0 && voxelCoord.z >= 0
                        && voxelCoord.x < dim.x && voxelCoord.y < dim.y && voxelCoord.z < dim.z) {
                    occupation = max(occupation, imageLoad(occupationVolumeImageIn, voxelCoord).r);
                }
            }
        }
    }

    imageStore(occupationVolumeImageOut, imageCoord, uvec4(occupation));
}

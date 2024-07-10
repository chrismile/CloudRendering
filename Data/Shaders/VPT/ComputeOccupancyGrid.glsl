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

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_vote : require
//#extension GL_EXT_debug_printf : enable

layout(local_size_x = SUBGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

#ifdef USE_NANOVDB
layout (binding = 0) readonly buffer NanoVdbBuffer {
    uint pnanovdb_buf_data[];
};
#else // USE_NANOVDB
layout (binding = 0) uniform sampler3D gridImage;
#endif // USE_NANOVDB

#if defined(ISOSURFACE_TYPE_GRADIENT)
layout(binding = 1) uniform sampler3D gradientImage;
#endif

#ifdef USE_TRANSFER_FUNCTION
layout(binding = 2) uniform sampler1DArray transferFunctionTexture;
#endif

layout (binding = 3, r8ui) uniform uimage3D occupancyGridImage;

layout (binding = 4) uniform Parameters {
    // Cloud properties.
    float voxelValueMin;
    float voxelValueMax;

    // Isosurfaces.
    float isoValue;
} parameters;

layout(push_constant) uniform PushConstants {
    uint blockSize, numBlocksTotal;
};

#ifdef USE_NANOVDB
float texelFetchRaw(in ivec3 coord) {
    // TODO
    pnanovdb_buf_t buf = pnanovdb_buf_t(0);
    pnanovdb_grid_handle_t gridHandle = pnanovdb_grid_handle_t(pnanovdb_address_null());
    vec3 posIndex = pnanovdb_grid_world_to_indexf(buf, gridHandle, pos);
    posIndex = floor(posIndex);
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, ivec3(posIndex));
    return pnanovdb_read_float(buf, address);
}
#else
float texelFetchRaw(in ivec3 coord) {
    return (texelFetch(gridImage, coord, 0).x - parameters.voxelValueMin) / (parameters.voxelValueMax - parameters.voxelValueMin);
}
#if defined(USE_ISOSURFACE_RENDERING) || defined(USE_ISOSURFACES)
#if defined(ISOSURFACE_TYPE_DENSITY)
#define isoImage gridImage
#elif defined(ISOSURFACE_TYPE_GRADIENT)
#define isoImage gradientImage
#endif
#define texelFetchIso(coord) (texelFetch(isoImage, coord, 0).x - parameters.voxelValueMin) / (parameters.voxelValueMax - parameters.voxelValueMin)
#endif // defined(USE_ISOSURFACE_RENDERING) || defined(USE_ISOSURFACES)
#endif

void main() {
    const uint localThreadIdx = gl_LocalInvocationID.x;
    const uint subgroupIdx = gl_WorkGroupID.x;
    const uvec3 occupancyGridSize = uvec3(imageSize(occupancyGridImage));
    //const uint numBlocksTotal = occupancyGridSize.x * occupancyGridSize.y * occupancyGridSize.z;
    if (subgroupIdx >= numBlocksTotal) {
        return;
    }

    const ivec3 gridSize = textureSize(gridImage, 0);
    uvec3 gridImageSubgroupIdx = uvec3(
            subgroupIdx % occupancyGridSize.x,
            (subgroupIdx / occupancyGridSize.x) % occupancyGridSize.y,
            subgroupIdx / (occupancyGridSize.x * occupancyGridSize.y)
    );

    const uint blockSizeLinearized = blockSize * blockSize * blockSize;
    bool isOccupied = true;
    for (uint blockLocalIdx = localThreadIdx; blockLocalIdx < blockSizeLinearized && !isOccupied; blockLocalIdx += SUBGROUP_SIZE) {
        uvec3 gridImageLocalOffset = uvec3(
                blockLocalIdx % blockSize,
                (blockLocalIdx / blockSize) % blockSize,
                blockLocalIdx / (blockSize * blockSize)
        );
        ivec3 gridIdx = ivec3(gridImageSubgroupIdx + gridImageLocalOffset);

        if (gridIdx.x < gridSize.x && gridIdx.y < gridSize.y && gridIdx.z < gridSize.z) {
            float densityRaw = texelFetchRaw(gridIdx);

#ifdef USE_TRANSFER_FUNCTION
            float density = texture(transferFunctionTexture, vec2(densityRaw, 0.0)).a;
#else
            float density = densityRaw;
#endif

#if defined(USE_ISOSURFACE_RENDERING) || defined(USE_ISOSURFACES)
#if defined(ISOSURFACE_TYPE_DENSITY)
            float isoDiff = densityRaw - parameters.isoValue;
#elif defined(ISOSURFACE_TYPE_GRADIENT)
            float isoDiff = texelFetchIso(gridIdx) - parameters.isoValue;
#endif
            isOccupied = isOccupied || !subgroupAllEqual(isoDiff < 0.0);
#endif

            isOccupied = isOccupied || subgroupAny(density > 1e-3);
        }
    }

    uint isOccupiedUint = isOccupied ? 1u : 0u;
    if (subgroupElect()) {
        imageStore(occupancyGridImage, ivec3(gridImageSubgroupIdx), uvec4(isOccupiedUint));
    }
}

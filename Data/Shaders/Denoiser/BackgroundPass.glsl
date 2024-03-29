/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Timm Knörle, Christoph Neuhauser
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

-- Compute

#version 460

layout(local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE) in;
//layout (binding = 0, rgba32f) uniform readonly image2D cloudOnlyImage;
layout (binding = 1, rgba32f) uniform image2D backgroundImage;
layout (binding = 2, rgba32f) uniform image2D resultImage;

void main() {
    ivec2 writePos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 outputImageSize = imageSize(resultImage);

    //if (writePos.x < outputImageSize.x && writePos.y < outputImageSize.y) {
        // get background color
        vec3 background = imageLoad(backgroundImage, writePos).xyz;

        // read cloud only texture
        vec4 cloudOnly = imageLoad(resultImage, writePos);

        // Accumulate result
        vec3 result = background.xyz * clamp(1. - cloudOnly.a, 0., 1.) + cloudOnly.xyz;
        //result = cloudOnly.xyz;
        imageStore(resultImage, writePos, vec4(result,1));
    //}

}

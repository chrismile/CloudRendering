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
        vec3 result = background.xyz * (1. - cloudOnly.a) + cloudOnly.xyz;
        //result = vec3(1,0,0);
        imageStore(resultImage, writePos, vec4(result,1));
    //}

}
